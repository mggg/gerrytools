import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.boxplot import BoxPlot


# =============================================
# == CONVERT DISTRIBUTION DATA TO DICTIONARY ==
# =============================================
class TestCategoricalDistributionBaseInit:
    def test_zero_group_width_raises_valueerror(self):
        with pytest.raises(ValueError, match="positive"):
            BoxPlot(group_width=0.0)

    def test_negative_group_width_raises_valueerror(self):
        with pytest.raises(ValueError, match="positive"):
            BoxPlot(group_width=-0.1)

    def test_group_width_greater_than_one_raises_valueerror(self):
        with pytest.raises(ValueError, match="<= 1.0"):
            BoxPlot(group_width=1.5)

    def test_zero_width_scale_raises_valueerror(self):
        with pytest.raises(ValueError, match="width_scale"):
            BoxPlot(width_scale=0.0)

    def test_negative_width_scale_raises_valueerror(self):
        with pytest.raises(ValueError, match="width_scale"):
            BoxPlot(width_scale=-0.1)

    def test_width_scale_greater_than_one_raises_valueerror(self):
        with pytest.raises(ValueError, match="width_scale"):
            BoxPlot(width_scale=1.5)

    def test_width_scale_one_is_valid(self):
        bp = BoxPlot(width_scale=1.0)
        assert bp.width_scale == 1.0

    def test_group_width_one_is_valid(self):
        bp = BoxPlot(group_width=1.0)
        assert bp.group_width == 1.0


# ===========================
# == LABEL SYNCHRONIZATION ==
# ===========================


class TestBoxPlotProperties:
    def test_group_width_getter_and_setter(self):
        bp = BoxPlot(group_width=0.5)
        assert bp.group_width == 0.5
        bp.group_width = 0.8
        assert bp.group_width == 0.8

    def test_width_scale_getter_and_setter(self):
        bp = BoxPlot(width_scale=0.6)
        assert bp.width_scale == 0.6
        bp.width_scale = 0.9
        assert bp.width_scale == 0.9


# =================================
# == BOXPLOT BUILD PRECONDITIONS ==
# =================================


class TestBoxPlotCategoryCenters:
    def test_category_centers_are_1_indexed(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2], "C": [3]})
        centers = bp._category_centers
        np.testing.assert_array_equal(centers, [1.0, 2.0, 3.0])

    def test_no_labels_returns_empty_array(self):
        bp = BoxPlot()
        assert len(bp._category_centers) == 0


# ==========================
# == BOXPLOT GROUP VLINES ==
# ==========================


class TestBoxPlotGroupVlines:
    def test_display_group_separators_false_disables(self):
        bp = BoxPlot()
        bp.display_group_separators(True)
        assert bp._include_group_vlines is True
        bp.display_group_separators(False)
        assert bp._include_group_vlines is False

    def test_update_group_vline_settings_enables(self):
        bp = BoxPlot()
        bp.update_group_vline_settings(linecolor="red")
        assert bp._include_group_vlines is True

    def test_clear_vertical_lines_and_bands_disables_vlines(self):
        bp = BoxPlot()
        bp.display_group_separators(True)
        bp.clear_verticals()
        assert bp._include_group_vlines is False
