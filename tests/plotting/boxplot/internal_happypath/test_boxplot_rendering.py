import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.boxplot import BoxPlot
from gerrytools.plotting.data.options import BoxPlotOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions


# =============================================
# == CONVERT DISTRIBUTION DATA TO DICTIONARY ==
# =============================================
class TestBoxPlotBuildPreconditions:
    def test_build_with_no_labels_raises_valueerror(self):
        bp = BoxPlot()
        with pytest.raises(ValueError, match="No labels"):
            bp.ax  # triggers build

    def test_build_with_no_datasets_raises_valueerror(self):
        bp = BoxPlot()
        # Manually set labels but add no data
        bp._labels = ["A", "B"]
        with pytest.raises(ValueError, match="No boxplot sets"):
            bp.ax

    def test_explicit_xlim_survives_default_category_ticks(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1, 2], "B": [2, 3], "C": [3, 4]})
        bp.set_xlim(1.5, 2.5)

        assert bp.ax.get_xlim() == pytest.approx((1.5, 2.5))


# ===============================
# == BOXPLOT POINTSET HANDLING ==
# ===============================


class TestBoxPlotPointset:
    def test_add_pointset_from_dict(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1, 2], "B": [3, 4]})
        bp.add_pointset({"A": 1.5, "B": 3.5}, name="Enacted")
        assert len(bp._pointset_data_list) == 1

    def test_add_pointset_from_list_uses_existing_labels(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        bp.add_pointset([1.5, 2.5])
        assert len(bp._pointset_data_list) == 1

    def test_add_pointset_from_list_without_labels_raises_when_no_data(self):
        bp = BoxPlot()
        with pytest.raises(ValueError, match="labels"):
            bp.add_pointset([1.5, 2.5])

    def test_add_pointset_length_mismatch_raises_valueerror(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        with pytest.raises(ValueError, match="length"):
            bp.add_pointset([1.5])  # only 1 value for 2 labels

    def test_add_pointset_from_series(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        ser = pd.Series({"A": 1.5, "B": 2.5})
        bp.add_pointset(ser)
        assert len(bp._pointset_data_list) == 1

    def test_add_pointset_from_dataframe_single_column(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        df = pd.DataFrame(
            {"val": [1.5, 2.5]},
            index=pd.Index(["A", "B"]),
        )
        bp.add_pointset(df)
        assert len(bp._pointset_data_list) == 1

    def test_add_pointset_from_dataframe_multi_column_no_column_raises(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        df = pd.DataFrame(
            {"v1": [1.5, 2.5], "v2": [3.5, 4.5]},
            index=pd.Index(["A", "B"]),
        )
        with pytest.raises(ValueError, match="exactly one"):
            bp.add_pointset(df)

    def test_add_pointset_from_dataframe_with_column_param(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        df = pd.DataFrame(
            {"v1": [1.5, 2.5], "v2": [3.5, 4.5]},
            index=pd.Index(["A", "B"]),
        )
        bp.add_pointset(df, column="v2")
        assert len(bp._pointset_data_list) == 1


# =========================
# == BOXPLOT AUTO-NAMING ==
# =========================


class TestBoxPlotLegendHandles:
    def test_legend_handles_include_boxplot_and_pointset(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1, 2]}, name="Ensemble")
        bp.add_pointset({"A": 1.5}, name="Enacted")
        handles = bp._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Ensemble" in labels
        assert "Enacted" in labels

    def test_named_lines_appear_in_legend(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1, 2]})
        bp.add_vertical_lines(1.5, name="Threshold")
        handles = bp._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Threshold" in labels

    def test_unnamed_lines_do_not_appear_in_legend(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1, 2]})
        bp.add_vertical_lines(1.5)  # no name
        handles = bp._legend_handles
        # Should only have the boxplot handle
        assert len(handles) == 1


class TestBoxPlotActualBuilds:
    """Smoke tests for supported build configurations that only promise not to raise."""

    def test_build_with_pointset_overlay(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
        bp.add_pointset({"A": 2.0, "B": 5.0}, name="Enacted")
        ax = bp.ax
        assert ax is not None

    def test_build_category_tick_labels_populated(self):
        bp = BoxPlot()
        bp.add_dataset({"Alpha": [1.0, 2.0], "Beta": [3.0, 4.0]})
        ax = bp.ax
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "Alpha" in tick_labels
        assert "Beta" in tick_labels

    def test_unlabeled_data_uses_numeric_tick_labels(self):
        bp = BoxPlot()
        bp.add_dataset([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        ax = bp.ax
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == ["0", "1", "2"]


# =======================================
# == EMPTY VALUE LISTS IN BOXPLOT DATA ==
# =======================================


class TestBoxPlotEmptyValueLists:
    def test_one_empty_category_is_skipped_silently(self):
        """Category with empty vals is skipped, others still drawn."""
        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [], "B": [1.0, 2.0, 3.0]})
        ax = bp.ax
        # Should build without error; at least one boxplot from "B"
        assert ax is not None

    def test_all_empty_categories_skips_entire_set(self):
        """All categories empty -> data_k empty -> entire set skipped."""
        bp = BoxPlot(legend=False)
        bp.add_dataset({"A": [], "B": []})
        ax = bp.ax
        # Should build without error, just draw nothing
        assert ax is not None


# ==================
# == FLIER ZORDER ==
# ==================
class TestFlierZorder:
    def test_explicit_flier_zorder_honored(self):
        bp = BoxPlot()
        samples = [float(value) for value in range(100)] + [500.0]
        bp.add_dataset(
            {"A": samples},
            showfliers=True,
            flier_options=PointMarkerOptions(zorder=10),
        )
        assert 10 in {line.get_zorder() for line in bp.ax.lines}

    def test_default_fliers_layer_with_the_set(self):
        bp = BoxPlot()
        samples = [float(value) for value in range(100)] + [500.0]
        bp.add_dataset({"A": samples}, showfliers=True, zorder=5)
        # Whiskers, caps, medians, and fliers all share the dataset zorder.
        assert {line.get_zorder() for line in bp.ax.lines} == {5}

    def test_flier_options_are_snapshotted_when_added(self):
        marker = PointMarkerOptions(marker="o")
        bp = BoxPlot()
        bp.add_dataset({"A": [1.0, 2.0]}, options=BoxPlotOptions(flier_options=marker))

        marker.marker = "x"

        assert bp._boxplot_data_list[0].style.flier_options.marker == "o"
