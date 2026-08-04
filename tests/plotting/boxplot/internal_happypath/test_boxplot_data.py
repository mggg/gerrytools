import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.boxplot import BoxPlot
from tests.plotting._typing_utils import as_any


# =============================================
# == CONVERT DISTRIBUTION DATA TO DICTIONARY ==
# =============================================
class TestConvertDistributionDataToDictionary:
    def test_dict_input_passthrough(self):
        result = BoxPlot._convert_distribution_data_to_dictionary({"A": [1, 2], "B": [3, 4]})
        assert result == {"A": [1, 2], "B": [3, 4]}

    def test_dict_keys_coerced_to_strings(self):
        result = BoxPlot._convert_distribution_data_to_dictionary(as_any({1: [10], 2: [20]}))
        assert "1" in result and "2" in result

    def test_dataframe_uses_columns_as_labels(self):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
        result = BoxPlot._convert_distribution_data_to_dictionary(df)
        assert set(result.keys()) == {"X", "Y"}
        assert result["X"] == [1.0, 2.0]

    def test_dataframe_drops_nan(self):
        df = pd.DataFrame({"X": [1.0, float("nan"), 3.0]})
        result = BoxPlot._convert_distribution_data_to_dictionary(df)
        assert result["X"] == [1.0, 3.0]

    def test_flat_list_with_labels_returns_single_entry_dict(self):
        result = BoxPlot._convert_distribution_data_to_dictionary(
            [10, 20, 30], category_labels=["A"]
        )
        assert "A" in result
        assert result["A"] == [10, 20, 30]

    def test_nested_list_with_labels(self):
        result = BoxPlot._convert_distribution_data_to_dictionary(
            [[1, 2], [3, 4]], category_labels=["A", "B"]
        )
        assert result == {"A": [1, 2], "B": [3, 4]}

    def test_flat_list_without_labels_is_single_numeric_category(self):
        result = BoxPlot._convert_distribution_data_to_dictionary([1, 2, 3])
        assert result == {"0": [1, 2, 3]}

    def test_nested_list_without_labels_auto_numbers_categories(self):
        result = BoxPlot._convert_distribution_data_to_dictionary([[1, 2], [3, 4], [5, 6]])
        assert result == {"0": [1, 2], "1": [3, 4], "2": [5, 6]}

    def test_2d_array_without_labels_auto_numbers_categories(self):
        result = BoxPlot._convert_distribution_data_to_dictionary(
            np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        assert result == {"0": [1.0, 2.0], "1": [3.0, 4.0]}

    def test_1d_array_is_a_single_category_like_a_flat_list(self):
        # Regression: a 1-D ndarray used to split into N single-value categories instead of
        # behaving like the equivalent flat list.
        result = BoxPlot._convert_distribution_data_to_dictionary(np.array([1.0, 2.0, 3.0]))
        assert result == {"0": [1.0, 2.0, 3.0]}

    def test_empty_list_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            BoxPlot._convert_distribution_data_to_dictionary([], category_labels=["A"])

    def test_labels_count_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="length"):
            BoxPlot._convert_distribution_data_to_dictionary(
                [[1, 2], [3, 4], [5, 6]], category_labels=["A", "B"]
            )

    def test_duplicate_labels_raise_valueerror(self):
        with pytest.raises(ValueError, match="unique"):
            BoxPlot._convert_distribution_data_to_dictionary(
                [[1, 2], [3, 4]], category_labels=["A", "A"]
            )

    def test_mapping_keys_that_stringify_alike_raise_valueerror(self):
        with pytest.raises(ValueError, match="unique"):
            BoxPlot._convert_distribution_data_to_dictionary(as_any({1: [1, 2], "1": [3, 4]}))

    def test_unsupported_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="dict"):
            BoxPlot._convert_distribution_data_to_dictionary(as_any(42))


# ======================================================
# == CATEGORICALDISTRIBUTIONPLOTBASE: INIT VALIDATION ==
# ======================================================


class TestLabelSynchronization:
    def test_first_dataset_defines_labels(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        assert bp._labels == ["A", "B"]

    def test_second_dataset_same_labels_succeeds(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        bp.add_dataset({"A": [3], "B": [4]})
        assert len(bp._boxplot_data_list) == 2

    def test_second_dataset_different_labels_raises_valueerror(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        with pytest.raises(ValueError, match="labels must match"):
            bp.add_dataset({"C": [3], "D": [4]})

    def test_add_extra_labels_merges_labels(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        bp.add_dataset({"A": [3], "B": [4], "C": [5]}, add_extra_labels=True)
        assert bp._labels is not None
        assert sorted(bp._labels) == ["A", "B", "C"]

    def test_add_extra_labels_preserves_original_order(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        bp.add_dataset({"C": [5], "A": [3]}, add_extra_labels=True)
        # Original order A, B maintained, with C appended
        assert bp._labels is not None
        assert bp._labels[:2] == ["A", "B"]


# ================================
# == BOXPLOT PROPERTY ACCESSORS ==
# ================================


class TestBoxPlotAutoNaming:
    def test_auto_name_for_boxplot_dataset(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1]})
        assert bp._boxplot_data_list[0].name == "Set 1"

    def test_explicit_name_for_boxplot_dataset(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1]}, name="Custom Name")
        assert bp._boxplot_data_list[0].name == "Custom Name"

    def test_auto_name_for_pointset(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1]})
        bp.add_pointset({"A": 1.5})
        assert bp._pointset_data_list[0].name == "Point Set 1"

    def test_explicit_name_for_pointset(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1]})
        bp.add_pointset({"A": 1.5}, name="Enacted Plan")
        assert bp._pointset_data_list[0].name == "Enacted Plan"

    def test_pointset_keys_that_stringify_alike_raise_valueerror(self):
        bp = BoxPlot()
        bp.add_dataset({"1": [1]})
        with pytest.raises(ValueError, match="unique"):
            bp.add_pointset(as_any({1: 1.5, "1": 2.5}))


# ===============================
# == BOXPLOT COLOR RESOLUTION ===
# ===============================


class TestBoxPlotColorResolution:
    def test_facecolor_none_resolves_to_none(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1, 2, 3]}, facecolor=None)
        assert bp._boxplot_data_list[0].style.facecolor == "none"

    def test_edgecolor_none_resolves_to_none_and_drops_edgewidth(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1, 2, 3]}, edgecolor=None)
        set_data = bp._boxplot_data_list[0]
        assert set_data.style.edgecolor == "none"
        assert set_data.style.edgewidth == 0.0

    def test_omitted_facecolor_uses_options_default(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1, 2, 3]})
        # The default "default_grey" resolves to its hex form.
        assert bp._boxplot_data_list[0].style.facecolor == "#5c676f"

    def test_explicit_facecolor_still_resolves(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1, 2, 3]}, facecolor="red")
        assert bp._boxplot_data_list[0].style.facecolor == "#ff0000"


# =========================
# == NAN-ONLY CATEGORIES ==
# =========================
class TestNanOnlyCategories:
    """A category with no finite samples is skipped (keeping its label slot) for dict and
    DataFrame input alike."""

    def test_dict_nan_only_category_skipped(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [float("nan"), float("nan")], "B": [1.0, 2.0, 3.0]})
        assert bp._labels == ["A", "B"]
        assert list(bp._boxplot_data_list[0].stats_dict.keys()) == ["B"]
        assert len(bp.ax.patches) == 1

    def test_dataframe_nan_only_category_skipped(self):
        bp = BoxPlot()
        bp.add_dataset(pd.DataFrame({"A": [np.nan, np.nan], "B": [1.0, 2.0]}))
        assert bp._labels == ["A", "B"]
        assert list(bp._boxplot_data_list[0].stats_dict.keys()) == ["B"]
        assert len(bp.ax.patches) == 1


# ==============================
# == BOXPLOT CATEGORY CENTERS ==
# ==============================
