import numpy as np
import pytest

from gerrytools.plotting.data.histogram import _HistogramData
from gerrytools.plotting.data.options import (
    BoxPlotOptions,
    HistogramOptions,
    SeaLevelLineOptions,
    ViolinPlotOptions,
    _FaceEdgeStyle,
)
from gerrytools.plotting.data.scatterplot import _ScatterData
from gerrytools.plotting.data.sealevel import _SeaLevelSetData
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from tests.plotting._typing_utils import as_any


# ============================
# == SHARED FACE/EDGE STYLE ==
# ============================
class TestFaceEdgeStyle:
    """The style validation lives once on _FaceEdgeStyle; every options class inherits it."""

    def test_negative_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            _FaceEdgeStyle(edgewidth=-1.0)

    def test_infinite_edgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            _FaceEdgeStyle(edgewidth=float("inf"))

    def test_edgecolor_none_resets_edgewidth_to_zero(self):
        style = _FaceEdgeStyle(edgecolor="none", edgewidth=3.0)
        assert style.edgewidth == 0.0

    def test_zorder_coerced_to_int(self):
        style = _FaceEdgeStyle(zorder=as_any(2.7))
        assert style.zorder == 2
        assert isinstance(style.zorder, int)

    def test_edgewidth_int_coerced_to_float(self):
        style = _FaceEdgeStyle(edgewidth=2)
        assert style.edgewidth == 2.0
        assert isinstance(style.edgewidth, float)

    def test_violin_options_inherit_the_shared_validation(self):
        with pytest.raises(ValueError, match="nonnegative"):
            ViolinPlotOptions(edgewidth=-1.0)
        assert ViolinPlotOptions(edgecolor="none", edgewidth=3.0).edgewidth == 0.0

    def test_histogram_options_inherit_the_shared_validation(self):
        with pytest.raises(ValueError, match="nonnegative"):
            HistogramOptions(edgewidth=-1.0)
        assert HistogramOptions(edgecolor="none", edgewidth=3.0).edgewidth == 0.0


class TestBoxPlotOptionsValidation:
    def test_default_construction(self):
        options = BoxPlotOptions(facecolor="blue")
        assert options.edgewidth == 0.8
        assert options.percentiles == (1.0, 99.0)
        assert options.showfliers is False

    def test_percentiles_out_of_range_raises_valueerror(self):
        with pytest.raises(ValueError, match="within"):
            BoxPlotOptions(percentiles=(-1, 50))

    def test_percentiles_high_greater_than_100_raises_valueerror(self):
        with pytest.raises(ValueError, match="within"):
            BoxPlotOptions(percentiles=(5, 101))

    def test_percentiles_low_equals_high_raises_valueerror(self):
        with pytest.raises(ValueError, match="low < high"):
            BoxPlotOptions(percentiles=(50, 50))

    def test_percentiles_low_greater_than_high_raises_valueerror(self):
        with pytest.raises(ValueError, match="low < high"):
            BoxPlotOptions(percentiles=(99, 1))


class TestSeaLevelSetData:
    def test_default_construction(self):
        sl = _SeaLevelSetData(name="test", scores_dict={"A": 0.5})
        assert sl.style.linewidth == 1.5
        assert sl.style.linecolor == "#000000"

    def test_style_validation_lives_in_the_options(self):
        with pytest.raises(ValueError, match="nonnegative"):
            _SeaLevelSetData(
                name="test",
                scores_dict={"A": 0.5},
                style=SeaLevelLineOptions(linewidth=-1.0),
            )


# ===================
# == HISTOGRAMDATA ==
# ===================


class TestHistogramData:
    def test_default_construction_with_valid_data(self):
        hd = _HistogramData(
            name="test",
            values=np.array([1.0, 2.0, 3.0]),
            weights=np.ones(3),
            style=HistogramOptions(),
        )
        assert hd.values.shape == (3,)
        assert hd.style.zorder == 2

    def test_weights_length_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="same length"):
            _HistogramData(
                name="test",
                values=np.array([1.0, 2.0]),
                weights=np.ones(3),
                style=HistogramOptions(),
            )

    def test_values_are_flattened(self):
        hd = _HistogramData(
            name="test",
            values=np.array([[1.0, 2.0], [3.0, 4.0]]),
            weights=np.ones(4),
            style=HistogramOptions(),
        )
        assert hd.values.shape == (4,)


class TestScatterData:
    def test_default_construction(self):
        sd = _ScatterData(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            name="test",
            marker_options=PointMarkerOptions(),
        )
        assert sd.x.shape == (2,)
        assert sd.name == "test"

    def test_shape_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="same shape"):
            _ScatterData(
                x=np.array([1.0, 2.0]),
                y=np.array([3.0]),
                name="test",
                marker_options=PointMarkerOptions(),
            )

    def test_multi_dimensional_arrays_raise_valueerror(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            _ScatterData(
                x=np.array([[1.0, 2.0]]),
                y=np.array([[3.0, 4.0]]),
                name="test",
                marker_options=PointMarkerOptions(),
            )

    def test_empty_arrays_raise_valueerror(self):
        with pytest.raises(ValueError, match="not be empty"):
            _ScatterData(
                x=np.array([]),
                y=np.array([]),
                name="test",
                marker_options=PointMarkerOptions(),
            )

    def test_none_label_is_valid(self):
        sd = _ScatterData(
            x=np.array([1.0]),
            y=np.array([2.0]),
            name=None,
            marker_options=PointMarkerOptions(),
        )
        assert sd.name is None


# ====================
# == SEATSVOTESDATA ==
# ====================
