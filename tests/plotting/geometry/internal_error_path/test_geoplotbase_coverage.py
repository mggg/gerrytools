import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, box

from gerrytools.plotting.geometry._labels import LabelOptions
from gerrytools.plotting.geometry.geoplot import GeoPlot
from gerrytools.plotting.geometry.geoplotbase import (
    _LabelRequest,
)
from gerrytools.plotting.mpl.label_text_options import LabelFontOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from tests.plotting._typing_utils import as_any


def _rect_gdf_with_crs(crs="EPSG:4326"):
    """Return a tiny 3-row GeoDataFrame of rectangles with the given CRS."""
    geoms = [
        box(0, 0, 1, 1),
        box(1, 0, 2, 1),
        box(0, 1, 1, 2),
    ]
    return GeoDataFrame(
        {"value": [10.0, 20.0, 30.0], "category": ["A", "B", "A"]},
        geometry=geoms,
        crs=crs,
    )


# ==========================
# == OUTLINE LAYER ERRORS ==
# ==========================
class TestAddOutlineLayerErrors:
    def test_dissolve_non_gdf_raises_typeerror(self, testing_gdf):
        """dissolve_column with GeoSeries raises TypeError."""
        gs = testing_gdf.geometry
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(TypeError, match="geo_source must be a GeoDataFrame"):
            plot.add_outline_layer(geo_source=gs, dissolve_column="district")

    def test_show_labels_without_dissolve_column_raises(self, testing_gdf):
        """show_labels=True without dissolve_column raises ValueError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="dissolve_column.*must be set"):
            plot.add_outline_layer(show_labels=True)

    def test_show_labels_without_dissolve_raises_before_registering(self, testing_gdf):
        """show_labels=True without dissolve_column raises up front; no layer is registered."""
        gs = testing_gdf.geometry
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        n_outlines = len(plot._outline_layers)
        with pytest.raises(ValueError, match="'dissolve_column' must be set"):
            plot.add_outline_layer(geo_source=gs, show_labels=True)
        assert len(plot._outline_layers) == n_outlines

    def test_show_labels_with_labelfont_options(self, testing_gdf, tmp_path):
        """show_labels=True with custom label_font_options uses them."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(
            dissolve_column="district",
            show_labels=True,
            label_options=LabelOptions(font_options=LabelFontOptions(fontsize=6, fontcolor="red")),
        )
        plot.save(str(tmp_path / "outline_custom_font.png"))
        assert (tmp_path / "outline_custom_font.png").exists()

    def test_show_labels_with_exclude_labels(self, testing_gdf, tmp_path):
        """show_labels=True with exclude_labels excludes them."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(
            dissolve_column="district",
            show_labels=True,
            label_options=LabelOptions(exclude=[0]),
        )
        plot.save(str(tmp_path / "outline_exclude.png"))
        assert (tmp_path / "outline_exclude.png").exists()


# ============================
# == HIGHLIGHT LAYER ERRORS ==
# ============================


class TestAddHighlightLayerErrors:
    def test_show_labels_no_label_column_raises(self, testing_gdf):
        """show_labels=True without label_column raises ValueError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="label_column"):
            plot.add_highlight_layer(
                geo_source=testing_gdf,
                show_labels=True,
                label_column=None,
            )

    def test_show_labels_no_geosource_raises(self, testing_gdf):
        """show_labels=True without geo_source raises ValueError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="geo_source"):
            plot.add_highlight_layer(
                show_labels=True,
                label_column="district",
                geo_source=None,
            )

    def test_show_labels_geoseries_geosource_raises(self, testing_gdf):
        """show_labels=True with GeoSeries geo_source raises TypeError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(TypeError, match="GeoDataFrame"):
            plot.add_highlight_layer(
                geo_source=testing_gdf.geometry,
                show_labels=True,
                label_column="district",
            )

    def test_highlight_geosource_none_uses_gdf(self, testing_gdf, tmp_path):
        """When geo_source=None, uses base gdf geometry."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer()  # geo_source=None
        plot.save(str(tmp_path / "highlight_no_src.png"))
        assert (tmp_path / "highlight_no_src.png").exists()

    def test_highlight_with_geometry_mask(self, testing_gdf, tmp_path):
        """geometry_mask filters highlight geometries."""
        mask = testing_gdf["district"] == 0
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer(geometry_mask=mask)
        plot.save(str(tmp_path / "highlight_mask.png"))
        assert (tmp_path / "highlight_mask.png").exists()

    def test_highlight_show_labels_with_mask(self, testing_gdf, tmp_path):
        """geometry_mask with show_labels applies mask to label_gdf."""
        mask = testing_gdf["district"] == 0
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer(
            geo_source=testing_gdf,
            show_labels=True,
            label_column="district",
            geometry_mask=mask,
        )
        plot.save(str(tmp_path / "highlight_mask_labels.png"))
        assert (tmp_path / "highlight_mask_labels.png").exists()

    def test_masks_are_positional_before_dissolve_and_labeling(self):
        gdf = GeoDataFrame(
            {"district": ["A", "B", "C", "D"]},
            geometry=[box(i, 0, i + 1, 1) for i in range(4)],
        )
        gdf.index = pd.Index([13, 12, 11, 10])
        mask = pd.Series([True, True, False, False], index=[10, 11, 12, 13])

        outline = GeoPlot(gdf, silent=True, default_outline=False)
        outline.add_outline_layer(
            geo_source=gdf,
            geometry_mask=mask,
            dissolve_column="district",
        )
        assert list(outline._outline_layers[-1].geometry_source["district"]) == ["A", "B"]

        highlight = GeoPlot(gdf, silent=True, default_outline=False)
        highlight.add_highlight_layer(
            geo_source=gdf,
            geometry_mask=mask,
            label_column="district",
            show_labels=True,
        )
        assert list(highlight._label_requests[-1].gdf["district"]) == ["A", "B"]

    def test_highlight_show_labels_with_custom_font(self, testing_gdf, tmp_path):
        """Custom label_font_options used when provided."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_highlight_layer(
            geo_source=testing_gdf,
            show_labels=True,
            label_column="district",
            label_options=LabelOptions(font_options=LabelFontOptions(fontsize=8)),
        )
        plot.save(str(tmp_path / "highlight_custom_font.png"))
        assert (tmp_path / "highlight_custom_font.png").exists()


# ==================
# == MARKER LAYER ==
# ==================


class TestAddMarkerLayer:
    def test_add_marker_layer_no_args_raises(self, testing_gdf):
        """Both args None raises ValueError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="Either"):
            plot.add_marker_layer()

    def test_add_marker_layer_both_args_raises(self, testing_gdf):
        """Both args provided raises ValueError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        pts = GeoSeries([Point(0, 0)])
        with pytest.raises(ValueError, match="Only one"):
            plot.add_marker_layer(
                points_geoseries=pts,
                latlon_list=[(0.0, 0.0)],
            )

    def test_add_marker_layer_with_lat_lon_list(self, testing_gdf, tmp_path):
        """lat/lon list path builds correctly."""
        gdf_crs = testing_gdf.copy().set_crs("EPSG:4326")
        plot = GeoPlot(gdf_crs, dpi=50, silent=True, target_crs="EPSG:4326")
        plot.add_marker_layer(
            latlon_list=[(40.0, -90.0), (41.0, -91.0)],
        )
        plot.save(str(tmp_path / "marker_latlon.png"))
        assert (tmp_path / "marker_latlon.png").exists()

    def test_add_marker_layer_with_geoseries(self, testing_gdf, tmp_path):
        """GeoSeries path builds correctly."""
        pts = GeoSeries([Point(2.0, 3.0), Point(5.0, 7.0)])
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_marker_layer(points_geoseries=pts, show_labels=False)
        plot.save(str(tmp_path / "marker_geoseries.png"))
        assert (tmp_path / "marker_geoseries.png").exists()

    def test_crs_bearing_marker_requires_plot_crs(self, testing_gdf):
        pts = GeoSeries([Point(-90.0, 40.0)])
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_marker_layer(
            points_geoseries=pts,
            input_crs="EPSG:4326",
            show_labels=False,
        )
        with pytest.raises(ValueError, match="CRS-bearing geometries"):
            plot.ax

    def test_projected_latlon_tuples_with_projected_input_crs(self, testing_gdf):
        """input_crs declares the CRS of the caller's coordinates for latlon_list too."""
        gdf = _rect_gdf_with_crs("EPSG:4326").to_crs("EPSG:3857")
        plot = GeoPlot(gdf, dpi=50, silent=True, target_crs="EPSG:3857")
        # A (y, x) pair already in EPSG:3857 map units: the marker must land exactly there.
        target_x, target_y = 55660.0, 111325.0
        plot.add_marker_layer(
            latlon_list=[(target_y, target_x)],
            input_crs="EPSG:3857",
            show_labels=False,
        )
        plot.ax
        rendered_marker = plot._ax.lines[-1]
        marker_x = np.asarray(rendered_marker.get_xdata(), dtype=float)
        marker_y = np.asarray(rendered_marker.get_ydata(), dtype=float)
        assert marker_x[0] == pytest.approx(target_x)
        assert marker_y[0] == pytest.approx(target_y)

    def test_add_marker_layer_with_all_options(self, testing_gdf, tmp_path):
        """Full marker layer with labels, custom marker options."""
        pts = GeoSeries([Point(2.0, 3.0)])
        marker_options = PointMarkerOptions(
            markerfacecolor="red",
            markerfacealpha=0.8,
            marker="^",
            markersize=4.0,
            markeredgecolor="black",
            markeredgealpha=1.0,
            markeredgewidth=0.5,
        )
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_marker_layer(
            points_geoseries=pts,
            labels=["Test"],
            show_labels=True,
            marker_options=marker_options,
        )
        plot.save(str(tmp_path / "marker_full.png"))
        assert (tmp_path / "marker_full.png").exists()


# =================
# == LABEL LAYER ==
# =================


class TestAddLabelLayer:
    def test_add_label_layer_no_args_raises(self, testing_gdf):
        """Both args None raises ValueError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="Either"):
            plot.add_label_layer()

    def test_add_label_layer_both_args_raises(self, testing_gdf):
        """Both args provided raises ValueError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        pts = GeoSeries([Point(0, 0)])
        with pytest.raises(ValueError, match="Only one"):
            plot.add_label_layer(
                points_geoseries=pts,
                latlon_list=[(0.0, 0.0)],
            )

    def test_add_label_layer_with_geoseries(self, testing_gdf, tmp_path):
        """GeoSeries path for add_label_layer builds correctly."""
        pts = GeoSeries([Point(2.0, 3.0), Point(5.0, 7.0)])
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_label_layer(points_geoseries=pts)
        plot.save(str(tmp_path / "label_layer_geoseries.png"))
        assert (tmp_path / "label_layer_geoseries.png").exists()

    def test_add_label_layer_with_lat_lon_list(self, testing_gdf, tmp_path):
        """lat/lon list path for add_label_layer."""
        gdf_crs = testing_gdf.copy().set_crs("EPSG:4326")
        plot = GeoPlot(gdf_crs, dpi=50, silent=True, target_crs="EPSG:4326")
        plot.add_label_layer(latlon_list=[(40.0, -90.0)])
        plot.save(str(tmp_path / "label_layer_latlon.png"))
        assert (tmp_path / "label_layer_latlon.png").exists()

    def test_add_label_layer_with_custom_labels(self, testing_gdf, tmp_path):
        """Custom labels list used instead of default numbering."""
        pts = GeoSeries([Point(2.0, 3.0)])
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_label_layer(
            points_geoseries=pts,
            labels=["MyLabel"],
            label_options=LabelOptions(font_options=LabelFontOptions(fontsize=6)),
        )
        plot.save(str(tmp_path / "label_layer_custom.png"))
        assert (tmp_path / "label_layer_custom.png").exists()


# ================
# == FOCUS AXES ==
# ================


class TestFocusAxesPaths:
    def test_focus_axes_with_geometry_mask(self, testing_gdf):
        """geometry_mask arg filters the geoseries."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        mask = testing_gdf["district"] == 0
        plot.focus_axes(geometry_mask=mask)
        minx, miny, maxx, maxy = testing_gdf.loc[mask].total_bounds
        assert plot.ax.get_xlim() == pytest.approx(
            (minx - 0.02 * (maxx - minx), maxx + 0.02 * (maxx - minx))
        )
        assert plot.ax.get_ylim() == pytest.approx(
            (miny - 0.02 * (maxy - miny), maxy + 0.02 * (maxy - miny))
        )

    def test_focus_axes_empty_mask_raises(self, testing_gdf):
        """All-False mask results in empty geoseries → ValueError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        mask = pd.Series([False] * len(testing_gdf), index=testing_gdf.index)
        with pytest.raises(ValueError, match=r"focus_axes\(\): no geometries"):
            plot.focus_axes(geometry_mask=mask)

    def test_focus_axes_with_crs_reprojection(self):
        """When geoseries.crs != target_crs, to_crs is called."""
        gdf = _rect_gdf_with_crs("EPSG:4326")
        plot = GeoPlot(gdf, dpi=50, silent=True, target_crs="EPSG:3857")
        plot.focus_axes()
        minx, miny, maxx, maxy = gdf.to_crs("EPSG:3857").total_bounds
        assert plot.ax.get_xlim() == pytest.approx(
            (minx - 0.02 * (maxx - minx), maxx + 0.02 * (maxx - minx))
        )
        assert plot.ax.get_ylim() == pytest.approx(
            (miny - 0.02 * (maxy - miny), maxy + 0.02 * (maxy - miny))
        )

    def test_focus_axes_tuple_pad(self, testing_gdf):
        """Tuple pad is split into (pad_x, pad_y)."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.focus_axes(pad=(0.01, 0.05))
        minx, miny, maxx, maxy = testing_gdf.total_bounds
        assert plot.ax.get_xlim() == pytest.approx(
            (minx - 0.01 * (maxx - minx), maxx + 0.01 * (maxx - minx))
        )
        assert plot.ax.get_ylim() == pytest.approx(
            (miny - 0.05 * (maxy - miny), maxy + 0.05 * (maxy - miny))
        )

    def test_focus_axes_invalid_pad_mode_raises(self, testing_gdf):
        """Invalid pad_mode raises ValueError."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        with pytest.raises(ValueError, match="pad_mode must be"):
            plot.focus_axes(pad_mode=as_any("invalid_mode"))

    def test_focus_axes_geosource_geoseries(self, testing_gdf):
        """GeoSeries as geo_source works without error."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.focus_axes(geo_source=testing_gdf.geometry)
        minx, miny, maxx, maxy = testing_gdf.total_bounds
        assert plot.ax.get_xlim() == pytest.approx(
            (minx - 0.02 * (maxx - minx), maxx + 0.02 * (maxx - minx))
        )
        assert plot.ax.get_ylim() == pytest.approx(
            (miny - 0.02 * (maxy - miny), maxy + 0.02 * (maxy - miny))
        )


# =====================
# == DEFERRED LABELS ==
# =====================


class TestDrawDeferredLabels:
    def test_labels_clip_with_renamed_geometry_column(self):
        """Label clipping must follow a renamed active geometry column.

        Regression: assigning dissolved["geometry"] was a no-op for renamed columns, so
        the representative point was computed on the unclipped geometry (here x = 5,
        outside the view) instead of inside the clipped region.
        """
        gdf = GeoDataFrame(
            {"region": ["wide"]},
            geometry=[box(0.0, 0.0, 10.0, 1.0)],
        )
        gdf.rename_geometry("shape", inplace=True)
        plot = GeoPlot(gdf, dpi=50, silent=True, default_outline=False)
        plot.add_outline_layer(dissolve_column="region", show_labels=True)
        plot.set_xlim(0.0, 1.0)
        plot.set_ylim(0.0, 1.0)
        _, positions = plot.get_label_positions()
        label_point = positions["wide"]
        assert 0.0 <= label_point.x <= 1.0
        assert 0.0 <= label_point.y <= 1.0

    def test_labels_crs_reprojection(self):
        """Labels are computed in the target CRS."""
        gdf = _rect_gdf_with_crs("EPSG:4326")
        plot = GeoPlot(gdf, dpi=50, silent=True, target_crs="EPSG:3857")
        plot.add_outline_layer(dissolve_column="category", show_labels=True)
        crs, positions = plot.get_label_positions()
        assert crs == "EPSG:3857"
        assert min(point.x for point in positions.values()) > 50_000

    def test_labels_skip_when_clip_empty(self, testing_gdf, tmp_path):
        """Labels outside the current view are clipped and skipped."""
        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        plot.add_outline_layer(dissolve_column="district", show_labels=True)
        # Set xlim/ylim to a region outside the data
        plot.set_xlim(1000, 2000)
        plot.set_ylim(1000, 2000)
        plot.save(str(tmp_path / "labels_skip.png"))
        assert (tmp_path / "labels_skip.png").exists()

    def test_label_format_fn_errors_propagate(self, testing_gdf, tmp_path):
        """A raising label_format_fn propagates instead of being silently swallowed."""

        def bad_fn(x):
            raise ValueError("intentional error")

        plot = GeoPlot(testing_gdf, dpi=50, silent=True)
        dissolved = GeoDataFrame(testing_gdf.dissolve(by="district").reset_index())
        req = _LabelRequest(
            gdf=dissolved,
            label_column="district",
            options=LabelOptions(),
            label_format_fn=bad_fn,
        )
        plot._label_requests.append(req)
        with pytest.raises(ValueError, match="intentional error"):
            plot.save(str(tmp_path / "label_format_fallback.png"))

    def test_duplicate_label_text_across_layers_uses_last_position(self):
        base = GeoDataFrame({"region": ["base"]}, geometry=[box(0, 0, 10, 10)])
        left = GeoDataFrame({"region": ["same"]}, geometry=[box(1, 1, 2, 2)])
        right = GeoDataFrame({"region": ["same"]}, geometry=[box(7, 7, 8, 8)])
        plot = GeoPlot(base, dpi=50, silent=True, default_outline=False)
        plot.add_highlight_layer("region", geo_source=left, show_labels=True)
        plot.add_highlight_layer("region", geo_source=right, show_labels=True)

        _, positions = plot.get_label_positions()

        assert positions["same"].equals(box(7, 7, 8, 8).representative_point())


# =====================
# == LABEL POSITIONS ==
# =====================


class TestGetLabelPositionsAsLatLong:
    def test_get_label_positions_as_lat_long(self):
        """as_lat_long=True converts label positions to EPSG:4326."""
        gdf = _rect_gdf_with_crs("EPSG:4326").to_crs("EPSG:3857")
        plot = GeoPlot(gdf, dpi=50, silent=True, target_crs="EPSG:3857")
        plot.add_outline_layer(dissolve_column="category", show_labels=True)
        crs_str, positions = plot.get_label_positions(as_lat_long=True)
        assert crs_str == "EPSG:4326"
        assert positions["A"].x == pytest.approx(0.5, abs=1e-4)
        assert positions["A"].y == pytest.approx(1.5, abs=1e-4)
        assert positions["B"].x == pytest.approx(1.5, abs=1e-4)
        assert positions["B"].y == pytest.approx(0.5, abs=1e-4)
