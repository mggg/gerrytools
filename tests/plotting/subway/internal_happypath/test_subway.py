"""Tests for subway sign plot."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pytest
from matplotlib import colors as mcolors

from gerrytools.plotting.other.subway import (
    SubwaySignOptions,
    _validate_subway_settings,
    subway_signs,
)
from tests.plotting._typing_utils import as_any

# ================
# == Validation ==
# ================


class TestSubwayValidation:
    """Cover _validate_subway_settings error paths."""

    def _default_options(self):
        return SubwaySignOptions()

    def test_invalid_orientation_raises(self):
        with pytest.raises(ValueError, match="orientation"):
            _validate_subway_settings(
                colors=["red"],
                labels=["A"],
                orientation=as_any("diagonal"),
                n_bands=None,
                max_items_per_band=None,
                sign_options=self._default_options(),
            )

    def test_both_n_bands_and_max_items_raises(self):
        with pytest.raises(ValueError, match="Only one"):
            _validate_subway_settings(
                colors=["red", "blue"],
                labels=["A", "B"],
                orientation="horizontal",
                n_bands=2,
                max_items_per_band=1,
                sign_options=self._default_options(),
            )

    def test_mismatched_label_color_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            _validate_subway_settings(
                colors=["red", "blue"],
                labels=["A"],
                orientation="horizontal",
                n_bands=None,
                max_items_per_band=None,
                sign_options=self._default_options(),
            )

    def test_empty_labels_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_subway_settings(
                colors=[],
                labels=[],
                orientation="horizontal",
                n_bands=None,
                max_items_per_band=None,
                sign_options=self._default_options(),
            )

    def test_bad_radius_raises(self):
        bad_options = SubwaySignOptions(radius=-1.0)
        with pytest.raises(ValueError, match="radius"):
            _validate_subway_settings(
                colors=["red"],
                labels=["A"],
                orientation="horizontal",
                n_bands=None,
                max_items_per_band=None,
                sign_options=bad_options,
            )

    def test_zero_n_bands_raises(self):
        with pytest.raises(ValueError, match="n_bands"):
            _validate_subway_settings(
                colors=["red"],
                labels=["A"],
                orientation="horizontal",
                n_bands=0,
                max_items_per_band=None,
                sign_options=self._default_options(),
            )

    def test_zero_max_items_per_band_raises(self):
        with pytest.raises(ValueError, match="max_items_per_band"):
            _validate_subway_settings(
                colors=["red"],
                labels=["A"],
                orientation="horizontal",
                n_bands=None,
                max_items_per_band=0,
                sign_options=self._default_options(),
            )


# =====================
# == Basic rendering ==
# =====================


class TestSubwayBasicRendering:
    def test_simple_call_returns_figure_and_axes(self):
        """Regression: the figure used to be created and abandoned with nothing returned."""
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

        figure, axes = subway_signs(
            colors=["red", "blue", "green"],
            labels=["A", "B", "C"],
        )
        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
        assert axes.figure is figure
        plt.close(figure)

    def test_supplied_axes_is_reused_without_creating_a_figure(self):
        import matplotlib.pyplot as plt

        supplied_figure, supplied_axes = plt.subplots()
        open_figures_before = len(plt.get_fignums())
        figure, axes = subway_signs(
            colors=["red", "blue"],
            labels=["A", "B"],
            ax=supplied_axes,
        )
        assert axes is supplied_axes
        assert figure is supplied_figure
        assert len(plt.get_fignums()) == open_figures_before
        plt.close(supplied_figure)

    def test_save_path_creates_file(self, tmp_path):
        """filepath should produce a file on disk."""
        out = str(tmp_path / "subway.png")
        subway_signs(
            colors=["red", "blue", "green"],
            labels=["A", "B", "C"],
            filepath=out,
        )
        assert Path(out).exists()

    def test_single_sign(self, tmp_path):
        """A single sign should render without error."""
        out = str(tmp_path / "single.png")
        subway_signs(colors=["#3366cc"], labels=["1"], filepath=out)
        assert Path(out).exists()

    def test_sign_options_accept_gerrytools_colors(self):
        import matplotlib.pyplot as plt

        figure, axes = subway_signs(
            colors=["red"],
            labels=["A"],
            sign_options=SubwaySignOptions(edgecolor="denim", fontcolor="denim"),
        )

        assert axes.patches[0].get_edgecolor() == pytest.approx(mcolors.to_rgba("#1560bd"))
        assert mcolors.to_rgba(axes.texts[0].get_color()) == pytest.approx(
            mcolors.to_rgba("#1560bd")
        )
        plt.close(figure)


# ====================
# == Layout options ==
# ====================


class TestSubwayLayoutOptions:
    def test_max_items_per_band(self, tmp_path):
        """max_items_per_band=2 distributes 5 items across multiple rows."""
        out = str(tmp_path / "max2.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple"],
            labels=["1", "2", "3", "4", "5"],
            max_items_per_band=2,
            filepath=out,
        )
        assert Path(out).exists()

    def test_max_items_per_band_larger_than_item_count_does_not_add_blank_space(self):
        figure, _ = subway_signs(
            colors=["red", "blue", "green"],
            labels=["1", "2", "3"],
            max_items_per_band=10,
        )

        assert figure.get_size_inches() == pytest.approx([2.1, 0.72])

    def test_n_bands(self, tmp_path):
        """n_bands=2 distributes 6 items across 2 rows."""
        out = str(tmp_path / "nbands2.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "yellow"],
            labels=["1", "2", "3", "4", "5", "6"],
            n_bands=2,
            filepath=out,
        )
        assert Path(out).exists()

    def test_default_single_band_horizontal(self, tmp_path):
        """Default layout should place all signs in one horizontal band."""
        out = str(tmp_path / "single_band.png")
        subway_signs(
            colors=["red", "blue", "green"],
            labels=["A", "B", "C"],
            orientation="horizontal",
            filepath=out,
        )
        assert Path(out).exists()

    def test_vertical_orientation_default(self, tmp_path):
        """Default vertical layout places all signs in one column."""
        out = str(tmp_path / "vertical.png")
        subway_signs(
            colors=["red", "blue", "green"],
            labels=["A", "B", "C"],
            orientation="vertical",
            filepath=out,
        )
        assert Path(out).exists()

    def test_vertical_with_max_items_per_band(self, tmp_path):
        """Vertical with max_items_per_band should create multiple columns."""
        out = str(tmp_path / "vertical_max.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple"],
            labels=["1", "2", "3", "4", "5"],
            orientation="vertical",
            max_items_per_band=3,
            filepath=out,
        )
        assert Path(out).exists()

    def test_horizontal_with_n_bands(self, tmp_path):
        """Horizontal with n_bands distributes into rows."""
        out = str(tmp_path / "horiz_nbands.png")
        subway_signs(
            colors=["red", "blue", "green", "orange"],
            labels=["A", "B", "C", "D"],
            orientation="horizontal",
            n_bands=2,
            filepath=out,
        )
        assert Path(out).exists()


# =================
# == Ragged edge ==
# =================


class TestSubwayRaggedEdge:
    def test_ragged_edge_last(self, tmp_path):
        """7 signs in 3-column layout with raggededge='last'."""
        out = str(tmp_path / "ragged_last.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="last"),
            filepath=out,
        )
        assert Path(out).exists()

    def test_ragged_edge_first(self, tmp_path):
        """7 signs in 3-column layout with raggededge='first'."""
        out = str(tmp_path / "ragged_first.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="first"),
            filepath=out,
        )
        assert Path(out).exists()

    def test_ragged_edge_vertical_last(self, tmp_path):
        """7 signs vertical 3-row layout with raggededge='last'."""
        out = str(tmp_path / "ragged_vert_last.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            orientation="vertical",
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="last"),
            filepath=out,
        )
        assert Path(out).exists()

    def test_ragged_edge_vertical_first(self, tmp_path):
        """7 signs vertical 3-row layout with raggededge='first'."""
        out = str(tmp_path / "ragged_vert_first.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            orientation="vertical",
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="first"),
            filepath=out,
        )
        assert Path(out).exists()


# ===========================
# == Reverse display order ==
# ===========================


class TestSubwayReverseOrder:
    def test_reverse_display_order_horizontal(self, tmp_path):
        """reverse_display_order=True with horizontal layout."""
        out = str(tmp_path / "reverse_horiz.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan"],
            labels=["1", "2", "3", "4", "5", "6"],
            reverse_display_order=True,
            filepath=out,
        )
        assert Path(out).exists()

    def test_reverse_display_order_vertical(self, tmp_path):
        """reverse_display_order=True with vertical layout."""
        out = str(tmp_path / "reverse_vert.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan"],
            labels=["1", "2", "3", "4", "5", "6"],
            orientation="vertical",
            reverse_display_order=True,
            filepath=out,
        )
        assert Path(out).exists()

    def test_reverse_display_order_ragged_last(self, tmp_path):
        """reverse_display_order with ragged last edge."""
        out = str(tmp_path / "reverse_ragged_last.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            max_items_per_band=3,
            reverse_display_order=True,
            sign_options=SubwaySignOptions(raggededge="last"),
            filepath=out,
        )
        assert Path(out).exists()

    def test_reverse_display_order_ragged_first(self, tmp_path):
        """reverse_display_order with ragged first edge."""
        out = str(tmp_path / "reverse_ragged_first.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "cyan", "magenta"],
            labels=["1", "2", "3", "4", "5", "6", "7"],
            max_items_per_band=3,
            reverse_display_order=True,
            sign_options=SubwaySignOptions(raggededge="first"),
            filepath=out,
        )
        assert Path(out).exists()


# ========================
# == Position assertions ==
# ========================


def _sign_grid_positions(**kwargs) -> dict[str, tuple[float, float]]:
    """Render signs and return label -> (grid_x, grid_y) from the drawn artists.

    grid_x counts columns left to right and grid_y counts rows top to bottom, in units of
    one grid step; ragged bands land on half-step fractions.
    """
    import matplotlib.pyplot as plt

    options = kwargs.get("sign_options") or SubwaySignOptions()
    radius = options.radius
    x_step = 2 * radius + (
        options.horizontalgap if options.horizontalgap is not None else 0.3 * radius
    )
    y_step = 2 * radius + (options.verticalgap if options.verticalgap is not None else 0.3 * radius)
    figure, axes = subway_signs(**kwargs)
    try:
        max_y = max(text.get_position()[1] for text in axes.texts)
        positions = {}
        for text in axes.texts:
            x, y = text.get_position()
            positions[text.get_text()] = (
                round((x - radius) / x_step, 6),
                round((max_y - y) / y_step, 6),
            )
        return positions
    finally:
        plt.close(figure)


SEVEN_COLORS = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]
SEVEN_LABELS = ["1", "2", "3", "4", "5", "6", "7"]


class TestSubwayPositions:
    def test_horizontal_ragged_last_centers_the_partial_last_row(self):
        positions = _sign_grid_positions(
            colors=SEVEN_COLORS,
            labels=SEVEN_LABELS,
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="last"),
        )
        assert positions["1"] == (0.0, 0.0)
        assert positions["2"] == (1.0, 0.0)
        assert positions["3"] == (2.0, 0.0)
        assert positions["4"] == (0.0, 1.0)
        assert positions["6"] == (2.0, 1.0)
        # The single leftover sign sits centered under the 3-wide rows.
        assert positions["7"] == (1.0, 2.0)

    def test_horizontal_ragged_first_centers_the_partial_first_row(self):
        positions = _sign_grid_positions(
            colors=SEVEN_COLORS,
            labels=SEVEN_LABELS,
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="first"),
        )
        assert positions["1"] == (1.0, 0.0)
        assert positions["2"] == (0.0, 1.0)
        assert positions["4"] == (2.0, 1.0)
        assert positions["5"] == (0.0, 2.0)
        assert positions["7"] == (2.0, 2.0)

    def test_vertical_ragged_last_centers_the_partial_last_column(self):
        positions = _sign_grid_positions(
            colors=SEVEN_COLORS,
            labels=SEVEN_LABELS,
            orientation="vertical",
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="last"),
        )
        assert positions["1"] == (0.0, 0.0)
        assert positions["2"] == (0.0, 1.0)
        assert positions["3"] == (0.0, 2.0)
        assert positions["4"] == (1.0, 0.0)
        assert positions["6"] == (1.0, 2.0)
        # The single leftover sign sits centered beside the 3-tall columns.
        assert positions["7"] == (2.0, 1.0)

    def test_vertical_ragged_first_centers_the_partial_first_column(self):
        positions = _sign_grid_positions(
            colors=SEVEN_COLORS,
            labels=SEVEN_LABELS,
            orientation="vertical",
            max_items_per_band=3,
            sign_options=SubwaySignOptions(raggededge="first"),
        )
        assert positions["1"] == (0.0, 1.0)
        assert positions["2"] == (1.0, 0.0)
        assert positions["4"] == (1.0, 2.0)
        assert positions["5"] == (2.0, 0.0)
        assert positions["7"] == (2.0, 2.0)

    def test_reverse_order_ragged_last_keeps_the_ragged_edge_in_place(self):
        positions = _sign_grid_positions(
            colors=SEVEN_COLORS,
            labels=SEVEN_LABELS,
            max_items_per_band=3,
            reverse_display_order=True,
            sign_options=SubwaySignOptions(raggededge="last"),
        )
        # The flat sequence 1..7 reverses to 7..1 and reflows: full rows [7,6,5] and
        # [4,3,2], with 1 centered in the ragged last row.
        assert positions["7"] == (0.0, 0.0)
        assert positions["6"] == (1.0, 0.0)
        assert positions["5"] == (2.0, 0.0)
        assert positions["4"] == (0.0, 1.0)
        assert positions["2"] == (2.0, 1.0)
        assert positions["1"] == (1.0, 2.0)

    def test_reverse_order_ragged_first_keeps_the_ragged_edge_in_place(self):
        positions = _sign_grid_positions(
            colors=SEVEN_COLORS,
            labels=SEVEN_LABELS,
            max_items_per_band=3,
            reverse_display_order=True,
            sign_options=SubwaySignOptions(raggededge="first"),
        )
        # Reversed flat sequence 7..1 with 7 centered in the ragged first row, then full
        # rows [6,5,4] and [3,2,1].
        assert positions["7"] == (1.0, 0.0)
        assert positions["6"] == (0.0, 1.0)
        assert positions["4"] == (2.0, 1.0)
        assert positions["3"] == (0.0, 2.0)
        assert positions["1"] == (2.0, 2.0)

    def test_overpartitioned_n_bands_collapses_blank_bands(self):
        # Regression: 5 items over n_bands=4 gives band_size=2, which only needs 3 bands;
        # the old layout kept 4, leaving a blank band and skipping ragged centering.
        positions = _sign_grid_positions(
            colors=SEVEN_COLORS[:5],
            labels=SEVEN_LABELS[:5],
            n_bands=4,
        )
        assert positions["1"] == (0.0, 0.0)
        assert positions["2"] == (1.0, 0.0)
        assert positions["3"] == (0.0, 1.0)
        assert positions["4"] == (1.0, 1.0)
        # The lone leftover sign sits centered under the 2-wide rows, on the last row:
        # exactly 3 bands, no blank fourth band.
        assert positions["5"] == (0.5, 2.0)
        assert max(grid_y for _, grid_y in positions.values()) == 2.0

    def test_reverse_order_full_grid_reverses_bands_and_items(self):
        # Reversing reverses the flat sequence 1..6 to 6..1 and re-bands it, reversing
        # within-band order as well as band order.
        positions = _sign_grid_positions(
            colors=SEVEN_COLORS[:6],
            labels=SEVEN_LABELS[:6],
            max_items_per_band=3,
            reverse_display_order=True,
        )
        assert positions["6"] == (0.0, 0.0)
        assert positions["4"] == (2.0, 0.0)
        assert positions["3"] == (0.0, 1.0)
        assert positions["1"] == (2.0, 1.0)

    def test_reverse_order_single_band_reverses_items(self):
        # Regression: the default single-band layout used to render 1, 2, 3 unchanged
        # because only band order was reversed, never within-band order.
        positions = _sign_grid_positions(
            colors=SEVEN_COLORS[:3],
            labels=SEVEN_LABELS[:3],
            reverse_display_order=True,
        )
        assert positions["3"] == (0.0, 0.0)
        assert positions["2"] == (1.0, 0.0)
        assert positions["1"] == (2.0, 0.0)


# ======================
# == Missing coverage ==
# ======================


class TestSubwayCoverageMissing:
    """Cover remaining uncovered branches: n_bands+vertical, invalid raggededge, bad color."""

    def test_n_bands_vertical_layout(self, tmp_path):
        """Vertical layouts also support `n_bands`."""
        out = str(tmp_path / "nbands_vert.png")
        subway_signs(
            colors=["red", "blue", "green", "orange", "purple", "yellow"],
            labels=["1", "2", "3", "4", "5", "6"],
            orientation="vertical",
            n_bands=2,
            filepath=out,
        )
        assert Path(out).exists()

    def test_invalid_raggededge_raises(self):
        """Invalid ragged-edge values raise `ValueError` up front, in validation."""
        import dataclasses

        opts = dataclasses.replace(SubwaySignOptions(), raggededge="invalid")
        with pytest.raises(ValueError, match="raggededge"):
            subway_signs(colors=["red"], labels=["A"], sign_options=opts)

    def test_invalid_color_raises(self):
        """Unconvertible colors raise `ValueError`."""
        with pytest.raises(ValueError, match="not_a_real_color_xyz"):
            subway_signs(
                colors=["not_a_real_color_xyz"],
                labels=["A"],
            )
