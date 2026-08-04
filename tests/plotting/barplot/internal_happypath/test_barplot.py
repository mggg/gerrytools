import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from gerrytools.plotting.data.barplot import BarPlot, _BarSetData
from gerrytools.plotting.data.options import BarPlotOptions


def _bars(ax: Axes) -> list[Rectangle]:
    """Collect the drawn (nonzero-height) bar rectangles from an axes."""
    return [p for p in ax.patches if isinstance(p, Rectangle) and p.get_height() > 0]


# ====================
# == HEIGHTS INPUT  ==
# ====================


class TestBarPlotHeightsInput:
    def test_add_bar_dataset_from_dict(self):
        bp = BarPlot()
        bp.add_dataset({"A": 1.0, "B": 2.0}, name="Enacted")
        assert bp._labels == ["A", "B"]
        assert bp._bar_data_list[0].heights_dict == {"A": 1.0, "B": 2.0}

    def test_add_bar_dataset_from_series(self):
        bp = BarPlot()
        bp.add_dataset(pd.Series({"A": 1.0, "B": 2.0}))
        assert bp._labels == ["A", "B"]

    def test_add_bar_dataset_from_list_with_labels(self):
        bp = BarPlot()
        bp.add_dataset([1.0, 2.0], labels=["A", "B"])
        assert bp._bar_data_list[0].heights_dict == {"A": 1.0, "B": 2.0}

    def test_add_bar_dataset_from_list_without_labels_raises(self):
        bp = BarPlot()
        with pytest.raises(ValueError, match="labels"):
            bp.add_dataset([1.0, 2.0])

    def test_add_bar_dataset_from_dataframe_with_column(self):
        bp = BarPlot()
        df = pd.DataFrame(
            {"v1": [1.0, 2.0], "v2": [3.0, 4.0]},
            index=pd.Index(["A", "B"]),
        )
        bp.add_dataset(df, column="v2")
        assert bp._bar_data_list[0].heights_dict == {"A": 3.0, "B": 4.0}

    def test_empty_heights_raises(self):
        bp = BarPlot()
        with pytest.raises(ValueError, match="heights is empty"):
            bp.add_dataset({})

    def test_nonfinite_height_raises(self):
        with pytest.raises(ValueError, match="finite"):
            _BarSetData(name="bad", heights_dict={"A": float("nan")}, style=BarPlotOptions())

    def test_rejected_dataset_does_not_claim_labels(self):
        bp = BarPlot()
        with pytest.raises(ValueError, match="finite"):
            bp.add_dataset({"A": 1.0, "Bad": float("nan")})

        bp.add_dataset({"A": 1.0, "B": 2.0})
        assert bp._labels == ["A", "B"]

    def test_mismatched_labels_raise_without_add_extra_labels(self):
        bp = BarPlot()
        bp.add_dataset({"A": 1.0, "B": 2.0})
        with pytest.raises(ValueError, match="must match existing labels"):
            bp.add_dataset({"A": 1.0, "C": 3.0})

    def test_add_extra_labels_merges(self):
        bp = BarPlot()
        bp.add_dataset({"A": 1.0, "B": 2.0})
        bp.add_dataset({"A": 1.0, "C": 3.0}, add_extra_labels=True)
        assert bp._labels == ["A", "B", "C"]


# ==================
# == COUNTS INPUT ==
# ==================


class TestBarPlotCountsInput:
    def test_counts_integer_values(self):
        bp = BarPlot()
        bp.add_counts_dataset([1, 1, 2, 3, 3, 3])
        assert bp._labels == ["1", "2", "3"]
        assert bp._bar_data_list[0].heights_dict == {"1": 2.0, "2": 1.0, "3": 3.0}

    def test_counts_float_values_use_compact_labels(self):
        bp = BarPlot()
        bp.add_counts_dataset([1.5, 1.5, 2.0])
        assert bp._labels == ["1.5", "2"]

    def test_counts_keep_values_that_differ_beyond_six_significant_digits(self):
        bp = BarPlot()
        bp.add_counts_dataset([1000000.0, 1000000.0, 1000000.4])

        heights = bp._bar_data_list[0].heights_dict
        assert heights == {"1000000": 2.0, "1000000.4": 1.0}

    def test_counts_from_series_drops_nonfinite(self):
        bp = BarPlot()
        bp.add_counts_dataset(pd.Series([1.0, np.nan, 1.0]))
        assert bp._bar_data_list[0].heights_dict == {"1": 2.0}

    def test_counts_empty_raises(self):
        bp = BarPlot()
        with pytest.raises(ValueError, match="finite"):
            bp.add_counts_dataset([np.nan])

    def test_counts_align_with_heights_dataset(self):
        bp = BarPlot()
        bp.add_counts_dataset([1, 2, 2])
        bp.add_dataset({"1": 5.0, "2": 6.0}, name="Reference")
        assert bp._labels == ["1", "2"]


# ===============
# == RENDERING ==
# ===============


class TestBarPlotRendering:
    def test_build_with_no_data_raises(self):
        bp = BarPlot()
        with pytest.raises(ValueError, match="No labels"):
            bp.ax  # triggers build

    def test_grouped_bar_positions_and_widths(self):
        bp = BarPlot(group_width=0.8, width_scale=1.0)
        bp.add_dataset({"A": 1.0, "B": 2.0})
        bp.add_dataset({"A": 3.0, "B": 4.0})
        patches = _bars(bp.ax)
        assert len(patches) == 4
        # Two sets: slot width 0.4, offsets -0.2/+0.2 around centers 1.0 and 2.0.
        lefts = sorted(p.get_x() for p in patches)
        expected_centers = [0.8, 1.2, 1.8, 2.2]
        assert lefts == pytest.approx([c - 0.2 for c in expected_centers])
        assert all(p.get_width() == pytest.approx(0.4) for p in patches)

    def test_category_names_are_x_tick_labels(self):
        bp = BarPlot()
        bp.add_dataset({"Alpha": 1.0, "Beta": 2.0})

        assert [label.get_text() for label in bp.ax.get_xticklabels()] == ["Alpha", "Beta"]

    def test_grouped_skips_missing_category(self):
        bp = BarPlot()
        bp.add_dataset({"A": 1.0, "B": 2.0})
        bp.add_dataset({"A": 3.0}, add_extra_labels=True)
        patches = _bars(bp.ax)
        assert len(patches) == 3

    def test_stacked_bars_accumulate_bottoms(self):
        bp = BarPlot(stacked=True, group_width=0.8, width_scale=1.0)
        bp.add_dataset({"A": 1.0, "B": 2.0})
        bp.add_dataset({"A": 3.0, "B": 4.0})
        patches = _bars(bp.ax)
        assert len(patches) == 4
        # All bars share the category center; the second set sits on top of the first.
        by_x: dict[float, list[Rectangle]] = {}
        for p in patches:
            by_x.setdefault(round(p.get_x(), 6), []).append(p)
        assert all(len(group) == 2 for group in by_x.values())
        for group in by_x.values():
            bottom_bar, top_bar = sorted(group, key=lambda p: p.get_y())
            assert top_bar.get_y() == pytest.approx(bottom_bar.get_height())
        assert all(p.get_width() == pytest.approx(0.8) for p in patches)

    def test_stacked_missing_category_contributes_zero(self):
        bp = BarPlot(stacked=True)
        bp.add_dataset({"A": 1.0, "B": 2.0})
        bp.add_dataset({"A": 3.0}, add_extra_labels=True)
        bars = [p for p in bp.ax.patches if isinstance(p, Rectangle)]

        assert [p.get_height() for p in bars] == pytest.approx([1.0, 2.0, 3.0, 0.0])
        assert [p.get_y() for p in bars] == pytest.approx([0.0, 0.0, 1.0, 2.0])


# ============
# == LEGEND ==
# ============


class TestBarPlotLegend:
    def test_legend_handles_include_bar_and_point_sets(self):
        bp = BarPlot()
        bp.add_dataset({"A": 1.0}, name="Ensemble")
        bp.add_pointset({"A": 1.5}, name="Enacted")
        labels = [h.get_label() for h in bp._legend_handles]
        assert "Ensemble" in labels
        assert "Enacted" in labels

    def test_datasets_are_autonamed(self):
        bp = BarPlot()
        bp.add_counts_dataset([1, 2])
        bp.add_dataset({"1": 5.0, "2": 6.0})
        names = [bar_set.name for bar_set in bp._bar_data_list]
        assert names == ["Set 1", "Set 2"]
