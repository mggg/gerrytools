import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from gerrytools.plotting.data.sealevel import SeaLevelPlot
from tests.plotting._typing_utils import as_any


# ==================
# == CONSTRUCTION ==
# ==================
class TestSeaLevelConstruction:
    def test_default_construction(self):
        sl = SeaLevelPlot()
        assert sl._labels is None
        assert sl.jitter_rng_seed is None

    def test_construction_with_seed(self):
        sl = SeaLevelPlot(jitter_rng_seed=42)
        assert sl.jitter_rng_seed == 42

    def test_construction_with_rng(self):
        rng = np.random.default_rng(99)
        sl = SeaLevelPlot(jitter_rng=rng)
        assert sl.jitter_rng_seed is None

    def test_seed_and_rng_raises_valueerror(self):
        with pytest.raises(ValueError, match="not both"):
            SeaLevelPlot(jitter_rng_seed=42, jitter_rng=np.random.default_rng(0))

    def test_seed_setter(self):
        sl = SeaLevelPlot()
        sl.jitter_rng_seed = 123
        assert sl.jitter_rng_seed == 123

    def test_seed_setter_none(self):
        sl = SeaLevelPlot(jitter_rng_seed=42)
        sl.jitter_rng_seed = None
        assert sl.jitter_rng_seed is None


# ======================================
# == CONVERT SCORE DATA TO DICTIONARY ==
# ======================================


class TestSeaLevelDataConversion:
    def test_dict_input(self):
        sl = SeaLevelPlot()
        result = sl._convert_score_data_to_dictionary({"A": 0.5, "B": 0.7})
        assert result == {"A": 0.5, "B": 0.7}

    def test_list_input_with_labels(self):
        sl = SeaLevelPlot()
        result = sl._convert_score_data_to_dictionary([0.5, 0.7], category_labels=["A", "B"])
        assert result == {"A": 0.5, "B": 0.7}

    def test_list_without_labels_raises_valueerror(self):
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="category_labels"):
            sl._convert_score_data_to_dictionary([0.5, 0.7])

    def test_list_length_mismatch_raises_valueerror(self):
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="does not match labels length"):
            sl._convert_score_data_to_dictionary([0.5, 0.7, 0.9], category_labels=["A", "B"])

    def test_series_input(self):
        sl = SeaLevelPlot()
        result = sl._convert_score_data_to_dictionary(pd.Series({"A": 0.5, "B": 0.7}))
        assert result == {"A": 0.5, "B": 0.7}

    def test_dataframe_input_with_row_index(self):
        sl = SeaLevelPlot()
        df = pd.DataFrame(
            {"A": [0.5, 0.6], "B": [0.7, 0.8]},
            index=pd.Index(["row1", "row2"]),
        )
        result = sl._convert_score_data_to_dictionary(df, df_row_index="row1")
        assert result == {"A": 0.5, "B": 0.7}

    def test_dataframe_without_row_index_raises_valueerror(self):
        sl = SeaLevelPlot()
        df = pd.DataFrame({"A": [0.5]})
        with pytest.raises(ValueError, match="df_row_index"):
            sl._convert_score_data_to_dictionary(df)

    def test_dataframe_missing_row_index_raises_valueerror(self):
        sl = SeaLevelPlot()
        df = pd.DataFrame(
            {"A": [0.5]},
            index=pd.Index(["row1"]),
        )
        with pytest.raises(ValueError, match="not found"):
            sl._convert_score_data_to_dictionary(df, df_row_index="bad_row")

    def test_non_finite_values_raise_valueerror(self):
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="finite"):
            sl._convert_score_data_to_dictionary({"A": float("nan")})

    def test_infinite_values_raise_valueerror(self):
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="finite"):
            sl._convert_score_data_to_dictionary({"A": float("inf")})

    def test_empty_dict_raises_valueerror(self):
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="empty"):
            sl._convert_score_data_to_dictionary({})

    def test_unsupported_type_raises_typeerror(self):
        sl = SeaLevelPlot()
        with pytest.raises(TypeError):
            sl._convert_score_data_to_dictionary(as_any(42))


# ===========================================
# == ADD SEALEVEL SET AND LABEL VALIDATION ==
# ===========================================


class TestAddSeaLevelSet:
    def test_first_set_defines_labels(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5, "B": 0.7})
        assert sl._labels == ["A", "B"]

    def test_second_set_same_labels_succeeds(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5, "B": 0.7})
        sl.add_dataset({"A": 0.3, "B": 0.6})
        assert len(sl._sealevel_data_list) == 2

    def test_second_set_different_labels_raises_valueerror(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5, "B": 0.7})
        with pytest.raises(ValueError, match="must match existing labels"):
            sl.add_dataset({"X": 0.3, "Y": 0.6})

    def test_auto_name_increments(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})
        sl.add_dataset({"A": 0.6})
        assert sl._sealevel_data_list[0].name == "Set 1"
        assert sl._sealevel_data_list[1].name == "Set 2"

    def test_explicit_name(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5}, name="Enacted Plan")
        assert sl._sealevel_data_list[0].name == "Enacted Plan"

    def test_markerfacecolor_defaults_to_linecolor(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5}, linecolor="red")
        ms = sl._sealevel_data_list[0].markersettings
        # markerfacecolor should have been derived from linecolor
        assert ms.markerfacecolor != "none"

    def test_markeredgecolor_defaults_to_markerfacecolor(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5}, linecolor="blue")
        ms = sl._sealevel_data_list[0].markersettings
        # Both face and edge should be derived from linecolor
        mpl_dict = ms.to_mpl_settings_dict()
        assert mpl_dict["markerfacecolor"] == mpl_dict["markeredgecolor"]

    def test_explicit_none_colors_remove_line_and_markers(self):
        sl = SeaLevelPlot()
        sl.add_dataset(
            {"A": 0.5},
            linecolor=None,
            markerfacecolor=None,
            markeredgecolor=None,
        )

        dataset = sl._sealevel_data_list[0]
        assert dataset.style.linecolor == "none"
        assert dataset.markersettings.markerfacecolor == "none"
        assert dataset.markersettings.markeredgecolor == "none"
        assert dataset.markersettings.markeredgewidth == 0.0


# ==========================
# == JITTER CONFIGURATION ==
# ==========================


class TestSeaLevelDataConversionDuplicateRow:
    def test_duplicate_row_index_raises_valueerror(self):
        sl = SeaLevelPlot()
        df = pd.DataFrame(
            {"A": [0.5, 0.6]},
            index=pd.Index(["row1", "row1"]),
        )
        with pytest.raises(ValueError, match="multiple rows"):
            sl._convert_score_data_to_dictionary(df, df_row_index="row1")


# ======================================
# == TICK LABEL MISMATCH RETURNS NONE ==
# ======================================


# ==========================
# == MARKER EDGE DEFAULTS ==
# ==========================


class TestSeaLevelMarkerEdgeResolution:
    def test_explicit_black_edge_in_marker_options_is_honored(self):
        # Regression: markeredgecolor != "black" was used as an "unset" sentinel, silently
        # overriding an explicitly requested black edge with the face color.
        from gerrytools.plotting.mpl.marker_options import PointMarkerOptions

        sl = SeaLevelPlot()
        sl.add_dataset(
            {"A": 0.5, "B": 0.7},
            marker_options=PointMarkerOptions(markerfacecolor="red", markeredgecolor="black"),
        )
        (dataset,) = sl._sealevel_data_list
        assert dataset.markersettings.markeredgecolor == "#000000"  # black, not the red face
        assert dataset.markersettings.markerfacecolor == "#ff0000"

    def test_default_edge_inherits_kwarg_face_color(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5, "B": 0.7}, markerfacecolor="red")
        (dataset,) = sl._sealevel_data_list
        assert dataset.markersettings.markeredgecolor == "#ff0000"

    def test_add_extra_labels_merges_new_categories(self):
        # The mismatch message advertises add_extra_labels=True, which now exists and works.
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5, "B": 0.7})
        with pytest.raises(ValueError, match="add_extra_labels"):
            sl.add_dataset({"A": 0.5, "C": 0.7})
        sl.add_dataset({"A": 0.5, "C": 0.7}, add_extra_labels=True)
        assert sl._labels == ["A", "B", "C"]


class TestSeaLevelListLabelFallback:
    def test_list_scores_fall_back_to_existing_labels(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 1.0, "B": 2.0})
        sl.add_dataset([3.0, 4.0])
        assert sl._sealevel_data_list[1].scores_dict == {"A": 3.0, "B": 4.0}

    def test_list_scores_without_any_labels_still_raises(self):
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="category_labels must be provided"):
            sl.add_dataset([1.0, 2.0])
