import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gerrytools.plotting.data.sealevel import SeaLevelPlot
from tests.plotting._typing_utils import as_any


# ==================
# == CONSTRUCTION ==
# ==================
class TestSeaLevelJitter:
    def _make_sealevel_with_labels(self):
        sl = SeaLevelPlot(jitter_rng_seed=42)
        sl.add_dataset({"A": 0.5, "B": 0.7, "C": 0.3})
        return sl

    def test_set_vertical_jitter_per_category(self):
        sl = self._make_sealevel_with_labels()
        sl.set_vertical_jitter(jitter={"A": 0.1, "B": 0.05})
        jitter = sl._maximum_vertical_jitter_per_category
        assert isinstance(jitter, dict)
        assert jitter["A"] == 0.1
        assert "C" not in jitter

    def test_set_vertical_jitter_all(self):
        sl = self._make_sealevel_with_labels()
        sl.set_vertical_jitter(0.1)
        assert sl._maximum_vertical_jitter_per_category == 0.1

    def test_set_horizontal_jitter_per_category(self):
        sl = self._make_sealevel_with_labels()
        sl.set_horizontal_jitter(jitter={"A": 0.05})
        jitter = sl._maximum_horizontal_jitter_per_category
        assert isinstance(jitter, dict)
        assert jitter["A"] == 0.05

    def test_set_horizontal_jitter_all(self):
        sl = self._make_sealevel_with_labels()
        sl.set_horizontal_jitter(0.2)
        assert sl._maximum_horizontal_jitter_per_category == 0.2

    def test_scalar_jitter_applies_to_labels_added_later(self):
        sl = SeaLevelPlot(jitter_rng_seed=42)
        sl.add_dataset({"A": 0.5})
        sl.set_vertical_jitter(0.1)
        sl.add_dataset({"B": 0.5}, name="later", add_extra_labels=True)

        later = next(line for line in sl.ax.lines if line.get_label() == "later")

        assert np.asarray(later.get_ydata())[0] != 0.5

    def test_jitter_before_labels_raises_valueerror(self):
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="No labels"):
            sl.set_vertical_jitter(0.1)

    def test_jitter_before_labels_horizontal_raises_valueerror(self):
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="No labels"):
            sl.set_horizontal_jitter(0.1)

    def test_jitter_extra_keys_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="Extra keys"):
            sl.set_vertical_jitter(jitter={"A": 0.1, "Z": 0.2})

    def test_negative_jitter_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_vertical_jitter(jitter={"A": -0.1})

    def test_infinite_jitter_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="finite"):
            sl.set_vertical_jitter(jitter={"A": float("inf")})

    def test_non_numeric_jitter_raises_typeerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(TypeError, match="real numbers"):
            sl.set_vertical_jitter(jitter=as_any({"A": "big"}))

    def test_non_dict_jitter_raises_typeerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(TypeError, match="dictionary"):
            sl.set_vertical_jitter(jitter=as_any([0.1, 0.2, 0.3]))

    def test_negative_jitter_all_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_vertical_jitter(-0.1)

    def test_infinite_jitter_all_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_vertical_jitter(float("inf"))

    def test_zero_jitter_is_valid(self):
        sl = self._make_sealevel_with_labels()
        sl.set_vertical_jitter(0.0)
        assert sl._maximum_vertical_jitter_per_category == 0.0


# =================================
# == FORMAT YLABELS AS FRACTIONS ==
# =================================


class TestFormatYLabelsAsFractions:
    def test_basic_fraction_labels(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})
        sl.format_ylabels_as_fractions(4)
        assert sl._yaxis.tick_locations == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert sl._yaxis.tick_labels == ["0/4", "1/4", "2/4", "3/4", "4/4"]

    def test_numpy_integer_denominator(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})

        sl.format_ylabels_as_fractions(np.int64(4))

        assert sl._yaxis.tick_labels == ["0/4", "1/4", "2/4", "3/4", "4/4"]

    def test_boolean_denominator_raises_typeerror(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})

        with pytest.raises(TypeError, match="integer"):
            sl.format_ylabels_as_fractions(True)

    def test_with_minimum_numerator(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})
        sl.format_ylabels_as_fractions(4, minimum_numerator=1)
        assert sl._yaxis.tick_locations is not None
        assert sl._yaxis.tick_locations[0] == 0.25

    def test_with_maximum_numerator(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})
        sl.format_ylabels_as_fractions(4, maximum_numerator=3)
        assert sl._yaxis.tick_locations is not None
        assert sl._yaxis.tick_locations[-1] == 0.75

    def test_non_int_denominator_raises_typeerror(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})
        with pytest.raises(TypeError, match="integer"):
            sl.format_ylabels_as_fractions(as_any(4.5))

    def test_zero_denominator_raises_valueerror(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})
        with pytest.raises(ValueError, match="positive"):
            sl.format_ylabels_as_fractions(0)

    def test_negative_denominator_raises_valueerror(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})
        with pytest.raises(ValueError, match="positive"):
            sl.format_ylabels_as_fractions(-4)

    def test_max_numerator_exceeds_denominator_raises_valueerror(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})
        with pytest.raises(ValueError, match="exceed"):
            sl.format_ylabels_as_fractions(4, maximum_numerator=5)

    def test_min_exceeds_max_raises_valueerror(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})
        with pytest.raises(ValueError, match="exceed"):
            sl.format_ylabels_as_fractions(4, minimum_numerator=3, maximum_numerator=2)

    def test_min_exceeds_denominator_when_max_is_none_raises_valueerror(self):
        sl = SeaLevelPlot()
        sl.add_dataset({"A": 0.5})
        with pytest.raises(ValueError, match="exceed"):
            sl.format_ylabels_as_fractions(4, minimum_numerator=5)


# =========================
# == BUILD PRECONDITIONS ==
# =========================


class TestSeaLevelHorizontalJitterValidation:
    def _make_sealevel_with_labels(self):
        sl = SeaLevelPlot(jitter_rng_seed=0)
        sl.add_dataset({"A": 0.5, "B": 0.7, "C": 0.3})
        return sl

    def test_non_dict_horizontal_jitter_raises_typeerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(TypeError, match="dictionary"):
            sl.set_horizontal_jitter(jitter=as_any([0.1, 0.2]))

    def test_non_real_horizontal_jitter_raises_typeerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(TypeError, match="real numbers"):
            sl.set_horizontal_jitter(jitter=as_any({"A": "big"}))

    def test_negative_horizontal_jitter_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_horizontal_jitter(jitter={"A": -0.1})

    def test_infinite_horizontal_jitter_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="finite"):
            sl.set_horizontal_jitter(jitter={"A": float("inf")})

    def test_no_labels_horizontal_per_category_raises_valueerror(self):
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="No labels"):
            sl.set_horizontal_jitter(jitter={"A": 0.1})

    def test_extra_keys_horizontal_per_category_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="Extra keys"):
            sl.set_horizontal_jitter(jitter={"A": 0.1, "Z": 0.2})

    def test_negative_horizontal_jitter_all_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_horizontal_jitter(-0.1)

    def test_infinite_horizontal_jitter_all_raises_valueerror(self):
        sl = self._make_sealevel_with_labels()
        with pytest.raises(ValueError, match="nonnegative"):
            sl.set_horizontal_jitter(float("inf"))

    def test_no_labels_vertical_per_category_raises_valueerror(self):
        """Direct call to set_vertical_jitter with no labels."""
        sl = SeaLevelPlot()
        with pytest.raises(ValueError, match="No labels"):
            sl.set_vertical_jitter(jitter={"A": 0.1})


class TestSeaLevelTickLabelMismatch:
    """Custom tick locations label category centers only."""

    def test_custom_tick_locations_label_category_centers(self):
        sl = SeaLevelPlot(legend=False)
        sl.add_dataset({"A": 0.5, "B": 0.6, "C": 0.7}, linecolor="black")
        # Set custom tick locations only, with a count that does not match the labels.
        sl.set_xticks(locations=[0.5, 1.0, 1.5, 2.0, 2.5])
        ax = sl.ax
        assert [tick.get_text() for tick in ax.get_xticklabels()] == [
            "",
            "A",
            "",
            "B",
            "",
        ]


# ============================================
# == FORMAT YLABELS AS FRACTIONS VALIDATION ==
# ============================================


class TestSeaLevelFractionLabelValidation:
    """Validation errors in `format_ylabels_as_fractions` are surfaced clearly."""

    def test_minimum_numerator_exceeds_denominator_raises(self):
        """A minimum numerator above the denominator is rejected."""
        sl = SeaLevelPlot(legend=False)
        sl.add_dataset({"A": 0.5}, linecolor="black")
        with pytest.raises(ValueError, match="minimum_numerator cannot exceed denominator"):
            sl.format_ylabels_as_fractions(10, minimum_numerator=11)

    def test_minimum_numerator_exceeds_explicit_maximum_raises(self):
        """A minimum numerator above the maximum numerator is rejected."""
        sl = SeaLevelPlot(legend=False)
        sl.add_dataset({"A": 0.5}, linecolor="black")
        with pytest.raises(ValueError, match="minimum_numerator cannot exceed maximum_numerator"):
            sl.format_ylabels_as_fractions(10, minimum_numerator=7, maximum_numerator=5)


# =======================================
# == FORMAT_YLABELS_AS_FRACTIONS TYPES ==
# =======================================


class TestFormatYLabelsAsFractionsTypeGuards:
    """Non-integer numerator inputs raise `TypeError`."""

    def test_float_minimum_numerator_raises_typeerror(self):
        from gerrytools.plotting.data.sealevel import SeaLevelPlot

        sl = SeaLevelPlot(legend=False)
        sl.add_dataset({"A": 0.5}, linecolor="black")
        with pytest.raises(TypeError, match="minimum_numerator must be an integer"):
            sl.format_ylabels_as_fractions(10, minimum_numerator=as_any(1.5))

    def test_float_maximum_numerator_raises_typeerror(self):
        from gerrytools.plotting.data.sealevel import SeaLevelPlot

        sl = SeaLevelPlot(legend=False)
        sl.add_dataset({"A": 0.5}, linecolor="black")
        with pytest.raises(TypeError, match="maximum_numerator must be an integer"):
            sl.format_ylabels_as_fractions(10, minimum_numerator=0, maximum_numerator=as_any(5.5))
