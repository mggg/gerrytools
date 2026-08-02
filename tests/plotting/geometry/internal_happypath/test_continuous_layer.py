import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.pyplot import get_cmap
from shapely.geometry import box

from gerrytools.plotting.geometry._layers._continuous import _ContinuousColorLayer
from tests.plotting._typing_utils import as_any


def make_layer(values, **kwargs):
    gdf = gpd.GeoDataFrame(
        {"value": values, "geometry": [box(i, 0, i + 1, 1) for i in range(len(values))]}
    )
    return _ContinuousColorLayer(geometry_source=gdf, column="value", **kwargs)


class TestBinsValidation:
    @pytest.mark.parametrize(
        ("bins", "error", "message"),
        [
            (0, ValueError, "at least 1"),
            (-3, ValueError, "at least 1"),
            ([5.0], ValueError, "at least 2 edges"),
            ([3.0, 1.0], ValueError, "strictly increasing"),
            ([1.0, 1.0], ValueError, "strictly increasing"),
            ([0.0, float("nan"), 2.0], ValueError, "finite"),
            ([0.0, float("inf")], ValueError, "finite"),
            ([0.0, "one"], TypeError, "float or int"),
            ([0.0, True], TypeError, "float or int"),
            (True, TypeError, "must be an int"),
            ("four", TypeError, "must be an int"),
        ],
    )
    def test_unusable_bins_are_rejected_at_construction(self, bins, error, message):
        with pytest.raises(error, match=message):
            make_layer([0.5, 1.5, 2.5], bins=bins)

    @pytest.mark.parametrize("bins", [1, 4, [0.0, 1.0, 2.0], None])
    def test_usable_bins_are_accepted(self, bins):
        assert make_layer([0.5, 1.5, 2.5], bins=bins).bins == bins

    @pytest.mark.parametrize("bounds", [{"vmin": 10.0}, {"vmax": -10.0}])
    def test_one_sided_bound_cannot_invert_data_range(self, bounds):
        layer = make_layer([1.0, 2.0, 3.0], **bounds)

        with pytest.raises(ValueError, match="less than"):
            _ = layer.color_series


class TestBinnedColorSeries:
    def test_colormap_instance_spans_its_range(self):
        cmap = get_cmap("viridis")
        layer = make_layer(
            [0.5, 1.5, 2.5],
            bins=[0.0, 1.0, 2.0, 3.0],
            colormap=cmap,
        )

        colors = layer.color_series

        assert colors.iloc[0][0] == to_hex(cmap(0.0))
        assert colors.iloc[-1][0] == to_hex(cmap(1.0))

    def test_value_on_terminal_break_with_larger_vmax_lands_in_last_bin(self):
        # Regression: a value equal to the last break with vmax beyond it escaped every guard
        # in the old get_loc/try/except lookup and raised UnboundLocalError.
        layer = make_layer([0.5, 1.5, 2.0], bins=[0.0, 1.0, 2.0], vmax=3.0)
        colors = layer.color_series
        assert colors.iloc[2] == colors.iloc[1]  # 2.0 joins 1.5 in the last bin
        assert colors.iloc[0] != colors.iloc[1]

    def test_interior_break_lands_in_the_bin_on_its_right(self):
        # Pins the closed="left" convention: 1.0 belongs to [1, 2), not [0, 1).
        layer = make_layer([0.5, 1.0, 1.5], bins=[0.0, 1.0, 2.0], vmax=3.0)
        colors = layer.color_series
        assert colors.iloc[1] == colors.iloc[2]
        assert colors.iloc[1] != colors.iloc[0]

    def test_vmin_vmax_replace_explicit_outer_breaks(self):
        layer = make_layer(
            [0.5, 1.5, 2.5],
            bins=[0.0, 1.0, 2.0],
            vmin=-1.0,
            vmax=3.0,
        )
        boundaries = layer._bin_boundaries(*layer._effective_bounds(layer._data_series()))
        assert boundaries.left.tolist() + [boundaries.right[-1]] == [-1.0, 1.0, 3.0]

    def test_binned_mappable_clamps_like_color_series(self):
        layer = make_layer([0.5, 1.5], bins=[0.0, 1.0, 2.0])
        mappable, _ = layer.mappable()
        assert mappable.norm(as_any(-1.0)) == 0
        assert mappable.norm(as_any(3.0)) == 1


class TestDuplicateIndexColorSeries:
    @pytest.mark.parametrize("bins", [None, 2])
    def test_duplicate_index_labels_get_their_own_colors(self, bins):
        # Regression: a dict keyed by index label collapsed duplicate labels onto the
        # last row's color; colors must be assigned positionally.
        gdf = gpd.GeoDataFrame(
            {"value": [0.0, 100.0], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
            index=pd.Index([0, 0]),
        )
        layer = _ContinuousColorLayer(geometry_source=gdf, column="value", bins=bins)
        colors = layer.color_series
        assert len(colors) == 2
        assert colors.iloc[0] != colors.iloc[1]

    def test_duplicate_index_with_geometry_mask_selects_positionally(self):
        # Regression: the trailing label-based reindex raised "cannot reindex on an axis
        # with duplicate labels" whenever a geometry_mask met a concat-style index.
        gdf = gpd.GeoDataFrame(
            {
                "value": [0.0, 50.0, 100.0],
                "geometry": [box(i, 0, i + 1, 1) for i in range(3)],
            },
            index=pd.Index([0, 0, 1]),
        )
        mask = pd.Series([True, False, True], index=gdf.index)
        masked = _ContinuousColorLayer(geometry_source=gdf, column="value", geometry_mask=mask)
        unmasked = _ContinuousColorLayer(geometry_source=gdf, column="value")
        colors = masked.color_series
        assert len(colors) == 2
        assert colors.iloc[0] == unmasked.color_series.iloc[0]
        assert colors.iloc[1] == unmasked.color_series.iloc[2]

    def test_geometry_and_colors_apply_reordered_mask_positionally(self):
        gdf = gpd.GeoDataFrame(
            {
                "value": [0.0, 50.0, 100.0],
                "geometry": [box(i, 0, i + 1, 1) for i in range(3)],
            },
            index=pd.Index(["a", "b", "c"]),
        )
        mask = pd.Series([True, True, False], index=["c", "b", "a"])
        layer = _ContinuousColorLayer(
            geometry_source=gdf,
            column="value",
            geometry_mask=mask,
        )

        assert layer.geometries.index.tolist() == ["a", "b"]
        assert layer.color_series.index.tolist() == ["a", "b"]


class TestConstantColumnBins:
    def test_constant_column_integer_bins_have_distinct_boundaries(self):
        # A constant column collapses lower == upper; the range is widened so integer
        # binning still produces distinct bin edges (and colorbar boundaries).
        layer = make_layer([5.0, 5.0, 5.0], bins=4)
        boundaries = layer._bin_boundaries(*layer._effective_bounds(layer._data_series()))
        edges = boundaries.left.tolist() + [boundaries.right[-1]]
        assert edges == sorted(set(edges))
        colors = layer.color_series
        assert colors.notna().all()
