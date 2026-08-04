from collections.abc import Sequence

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.boxplot import BoxPlot, BoxPlotStats, _BoxPlotSetData
from tests.plotting._typing_utils import as_any


def _stats(
    median: float = 0.5,
    spread: float = 0.1,
    *,
    mean: float | None = None,
    fliers: Sequence[float] = (),
) -> BoxPlotStats:
    """Build a valid BoxPlotStats centered on ``median`` with a symmetric spread."""
    return BoxPlotStats(
        median=median,
        lower_quartile=median - spread,
        upper_quartile=median + spread,
        lower_whisker=median - 2 * spread,
        upper_whisker=median + 2 * spread,
        mean=mean,
        fliers=fliers,
    )


# ==================
# == BOXPLOTSTATS ==
# ==================


class TestBoxPlotStatsValidation:
    def test_valid_construction_coerces_to_float(self):
        stats = BoxPlotStats(
            median=1, lower_quartile=0, upper_quartile=2, lower_whisker=-1, upper_whisker=3
        )
        assert isinstance(stats.median, float)
        assert stats.fliers == ()
        assert stats.mean is None

    def test_bad_ordering_raises(self):
        with pytest.raises(ValueError, match="lower_whisker <= lower_quartile"):
            BoxPlotStats(
                median=0.5,
                lower_quartile=0.6,  # > median
                upper_quartile=0.7,
                lower_whisker=0.3,
                upper_whisker=0.8,
            )

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            BoxPlotStats(
                median=float("nan"),
                lower_quartile=0.4,
                upper_quartile=0.6,
                lower_whisker=0.3,
                upper_whisker=0.7,
            )

    def test_non_finite_mean_raises(self):
        with pytest.raises(ValueError, match="mean must be a finite"):
            _stats(mean=float("inf"))

    def test_equal_values_are_allowed(self):
        # A degenerate distribution where everything collapses to one value.
        stats = BoxPlotStats(
            median=1.0, lower_quartile=1.0, upper_quartile=1.0, lower_whisker=1.0, upper_whisker=1.0
        )
        assert stats.median == 1.0

    def test_fliers_coerced_to_float_tuple(self):
        stats = _stats(fliers=[1, 2, 3])
        assert stats.fliers == (1.0, 2.0, 3.0)

    def test_to_bxp_dict_omits_mean_when_absent(self):
        d = _stats().to_bxp_dict("A")
        assert d["label"] == "A"
        assert {"med", "q1", "q3", "whislo", "whishi", "fliers"} <= set(d)
        assert "mean" not in d

    def test_to_bxp_dict_includes_mean_when_present(self):
        d = _stats(mean=0.5).to_bxp_dict("A")
        assert d["mean"] == 0.5


# =======================================
# == CONVERT STATS INPUT TO DICTIONARY ==
# =======================================


class TestConvertStatsToDictionary:
    def test_mapping_of_dataclasses_passthrough(self):
        result = BoxPlot._convert_boxplot_stats_to_dictionary({"A": _stats(), "B": _stats(1.5)})
        assert set(result) == {"A", "B"}
        assert all(isinstance(v, BoxPlotStats) for v in result.values())

    def test_mapping_of_plain_dicts(self):
        result = BoxPlot._convert_boxplot_stats_to_dictionary(
            {
                "A": {
                    "median": 0.5,
                    "lower_quartile": 0.4,
                    "upper_quartile": 0.6,
                    "lower_whisker": 0.3,
                    "upper_whisker": 0.7,
                }
            }
        )
        assert isinstance(result["A"], BoxPlotStats)
        assert result["A"].median == 0.5

    def test_dataframe_index_as_categories(self):
        df = pd.DataFrame(
            {
                "median": [0.5, 1.5],
                "lower_quartile": [0.4, 1.4],
                "upper_quartile": [0.6, 1.6],
                "lower_whisker": [0.3, 1.2],
                "upper_whisker": [0.7, 1.8],
            },
            index=pd.Index(["A", "B"]),
        )
        result = BoxPlot._convert_boxplot_stats_to_dictionary(df)
        assert set(result) == {"A", "B"}
        assert result["B"].median == 1.5

    def test_dataframe_nan_mean_becomes_none(self):
        df = pd.DataFrame(
            {
                "median": [0.5, 1.5],
                "lower_quartile": [0.4, 1.4],
                "upper_quartile": [0.6, 1.6],
                "lower_whisker": [0.3, 1.2],
                "upper_whisker": [0.7, 1.8],
                "mean": [0.5, float("nan")],
            },
            index=pd.Index(["A", "B"]),
        )
        result = BoxPlot._convert_boxplot_stats_to_dictionary(df)
        assert result["A"].mean == 0.5
        assert result["B"].mean is None

    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError, match="missing required"):
            BoxPlot._convert_boxplot_stats_to_dictionary(
                {"A": {"median": 0.5, "lower_quartile": 0.4}}
            )

    def test_unsupported_value_type_raises(self):
        with pytest.raises(TypeError, match="must be a BoxPlotStats"):
            BoxPlot._convert_boxplot_stats_to_dictionary(as_any({"A": 5}))

    def test_unsupported_container_raises(self):
        with pytest.raises(TypeError, match="Mapping"):
            BoxPlot._convert_boxplot_stats_to_dictionary(as_any(42))


# =================================
# == ADD STATS DATASETS BEHAVIOR ==
# =================================


class TestAddBoxplotStatsDatasets:
    def test_appends_stats_set(self):
        bp = BoxPlot()
        bp.add_stats_dataset({"A": _stats(), "B": _stats(1.5)})
        assert len(bp._boxplot_data_list) == 1
        assert isinstance(bp._boxplot_data_list[0], _BoxPlotSetData)

    def test_defines_labels(self):
        bp = BoxPlot()
        bp.add_stats_dataset({"A": _stats(), "B": _stats(1.5)})
        assert bp._labels == ["A", "B"]

    def test_auto_name_shares_counter_with_raw_sets(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1], "B": [2]})
        bp.add_stats_dataset({"A": _stats(), "B": _stats(1.5)})
        assert bp._boxplot_data_list[1].name == "Set 2"

    def test_explicit_name(self):
        bp = BoxPlot()
        bp.add_stats_dataset({"A": _stats()}, name="Precomputed")
        assert bp._boxplot_data_list[0].name == "Precomputed"

    def test_empty_stats_raises(self):
        bp = BoxPlot()
        with pytest.raises(ValueError, match="at least one category"):
            bp.add_stats_dataset({})

    def test_mismatched_labels_raise(self):
        bp = BoxPlot()
        bp.add_stats_dataset({"A": _stats(), "B": _stats(1.5)})
        with pytest.raises(ValueError, match="labels must match"):
            bp.add_stats_dataset({"C": _stats()})

    def test_styling_kwargs_override_options(self):
        bp = BoxPlot()
        bp.add_stats_dataset({"A": _stats()}, facecolor="red", edgewidth=2.0)
        set_data = bp._boxplot_data_list[0]
        assert set_data.style.facecolor == "#ff0000"
        assert set_data.style.edgewidth == 2.0

    def test_facecolor_none_resolves_to_none(self):
        bp = BoxPlot()
        bp.add_stats_dataset({"A": _stats()}, facecolor=None)
        assert bp._boxplot_data_list[0].style.facecolor == "none"

    def test_edgecolor_none_resolves_to_none_and_drops_edgewidth(self):
        bp = BoxPlot()
        bp.add_stats_dataset({"A": _stats()}, edgecolor=None)
        set_data = bp._boxplot_data_list[0]
        assert set_data.style.edgecolor == "none"
        assert set_data.style.edgewidth == 0.0


# ===================
# == ACTUAL BUILDS ==
# ===================


class TestStatsBuilds:
    def test_build_single_stats_set(self):
        bp = BoxPlot(legend=False)
        bp.add_stats_dataset({"A": _stats(), "B": _stats(1.5)})
        assert bp.ax is not None

    def test_medians_render_at_precomputed_values(self):
        bp = BoxPlot(legend=False)
        bp.add_stats_dataset({"A": _stats(median=0.55), "B": _stats(median=1.55)})
        ax = bp.ax
        # Median lines are horizontal (two equal y-values).
        horizontal_ys = set()
        for line in ax.get_lines():
            ydata = np.asarray(line.get_ydata(), dtype=float)
            if len(ydata) == 2 and ydata[0] == ydata[1]:
                horizontal_ys.add(round(float(ydata[0]), 3))
        assert 0.55 in horizontal_ys
        assert 1.55 in horizontal_ys

    def test_build_mixed_raw_and_stats(self):
        bp = BoxPlot()
        bp.add_dataset({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, name="Raw")
        bp.add_stats_dataset({"A": _stats(), "B": _stats(1.5)}, name="Stats")
        assert bp.ax is not None
        assert len(bp._boxplot_data_list) == 2

    def test_build_with_fliers_shown(self):
        bp = BoxPlot(legend=False)
        bp.add_stats_dataset({"A": _stats(fliers=[0.95, 0.99])}, showfliers=True)
        assert bp.ax is not None

    def test_build_with_means(self):
        bp = BoxPlot(legend=False)
        bp.add_stats_dataset({"A": _stats(mean=0.5), "B": _stats(1.5, mean=1.5)})
        assert bp.ax is not None

    def test_category_without_stats_is_skipped(self):
        bp = BoxPlot()
        # First set defines labels A, B; second stats set covers only A.
        bp.add_dataset({"A": [1.0], "B": [2.0]})
        bp.add_stats_dataset({"A": _stats()}, add_extra_labels=True)
        assert bp.ax is not None

    def test_stats_set_appears_in_legend(self):
        bp = BoxPlot()
        bp.add_stats_dataset({"A": _stats()}, name="Ensemble Summary")
        labels = [h.get_label() for h in bp._legend_handles]
        assert "Ensemble Summary" in labels


class TestCoerceFliersFromArray:
    def test_numpy_array_fliers_in_mapping(self):
        result = BoxPlot._convert_boxplot_stats_to_dictionary(
            {
                "A": {
                    "median": 0.5,
                    "lower_quartile": 0.4,
                    "upper_quartile": 0.6,
                    "lower_whisker": 0.3,
                    "upper_whisker": 0.7,
                    "fliers": np.array([0.9, 0.95]),
                }
            }
        )
        assert result["A"].fliers == (0.9, 0.95)
