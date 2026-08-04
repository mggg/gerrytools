import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.patches import Rectangle

from gerrytools.plotting.data.histogram import Histogram


# ===============================
# == CONSTRUCTION AND DEFAULTS ==
# ===============================
class TestHistogramActualBuilds:
    """Smoke tests for supported draw configurations that only promise not to raise."""

    def test_build_stack_histogram(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0], histtype="stack")
        h.add_dataset([4.0, 5.0, 6.0], histtype="stack")
        ax = h.ax
        assert ax is not None

    def test_build_grouped_histogram(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0], histtype="grouped")
        h.add_dataset([4.0, 5.0, 6.0], histtype="grouped")
        ax = h.ax
        assert ax is not None

    def test_build_outline_histogram(self):
        h = Histogram()
        h.add_dataset(
            [1.0, 2.0, 3.0],
            histtype="outline",
            facecolor="none",
            edgecolor="black",
            edgewidth=1.0,
        )
        ax = h.ax
        assert ax is not None

    def test_build_with_explicit_bins(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0])
        h.set_bins(5)
        ax = h.ax
        assert ax is not None

    def test_build_with_binwidth(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0])
        h.set_bin_widths(0.5)

        np.testing.assert_allclose(h._compute_bins(), [1.0, 1.5, 2.0, 2.5, 3.0])

    def test_build_with_points_above(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0, 4.0])
        h.set_bins([1.0, 2.0, 3.0, 4.0])
        h.add_points_above(4.0, name="Threshold")
        ax = h.ax

        y_positions = np.asarray(ax.lines[-1].get_ydata(), dtype=float)
        assert y_positions[0] > 2.0

    def test_build_with_centered_bin_alignment(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0])
        h.center_bars()
        ax = h.ax
        assert ax is not None

    def test_build_with_grid(self):
        h = Histogram()
        h.display_grid(True)
        h.add_dataset([1.0, 2.0, 3.0])
        ax = h.ax
        assert ax is not None

    def test_build_with_weights(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0], weights=[1.0, 2.0, 3.0])
        ax = h.ax
        assert ax is not None

    def test_build_with_point_outside_bin_range(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0])
        h.add_points_above([10.0, 11.0], name="OutOfRange", y_offset=0.5)
        ax = h.ax
        y_positions = np.asarray(ax.lines[-1].get_ydata(), dtype=float)

        assert y_positions[0] > 0.5
        assert y_positions[1] > y_positions[0]

    def test_build_stacked_with_point_shows_stacked_height(self):
        h = Histogram()
        h.set_bins([1.0, 2.0, 3.0])
        h.add_dataset([1.0, 1.5], histtype="stack")
        h.add_dataset([1.0, 1.5], histtype="stack")
        h.add_points_above(1.2, name="Plan")
        ax = h.ax

        y_positions = np.asarray(ax.lines[-1].get_ydata(), dtype=float)
        assert y_positions[0] > 4.0


class TestHistogramCenterOnBinEdgeErrors:
    """Non-uniform bins cannot be combined with centered bin labels."""

    def test_non_uniform_bins_with_centering_raises(self):
        h = Histogram()
        h.display_warnings(False)
        h.set_bins([0, 1, 3, 6])
        h.center_bars()
        h.add_dataset([0.5, 1.5, 4.0])
        with pytest.raises(ValueError, match="Cannot center histogram"):
            h.ax


# ===================================
# == GROUPED/STACK DEFAULT EDGES ==
# ===================================


class TestHistogramGroupedEdges:
    """Grouped and stack modes support the default visible bar edges."""

    def test_grouped_default_edges_do_not_warn(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0], histtype="grouped")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            h.ax
        assert not [x for x in w if issubclass(x.category, UserWarning)]


# ==========================================
# == OUTLINE HISTOGRAM WITH CENTERED BINS ==
# ==========================================


class TestHistogramOutlineCenteredBins:
    """Outline histograms still build when centered bins are enabled."""

    def test_outline_histogram_with_centered_bins(self):
        h = Histogram()
        h.display_warnings(False)
        h.center_bars()
        h.add_dataset(
            [1.0, 1.0, 2.0, 2.0, 3.0],
            histtype="outline",
            edgecolor="black",
            edgewidth=1.0,
        )
        ax = h.ax
        assert ax is not None


# ============================================
# == POINTS ABOVE WITH centered_on_bin=True ==
# ============================================


class TestHistogramPointsAboveCentered:
    """Centered bins adjust the point-above placement path."""

    def test_points_above_centered_on_bin(self):
        h = Histogram()
        h.display_warnings(False)
        h.add_dataset([1.0, 1.0, 2.0, 2.0, 3.0])
        h.add_points_above([1.0, 2.0, 3.0], centered_on_bin=True)
        ax = h.ax
        assert ax is not None


# =======================
# == DEGENERATE MARKER ==
# =======================


class TestMarkerClearanceZeroHeightPath:
    """Zero-height marker paths still compute marker clearance safely."""

    def test_horizontal_bar_marker_hits_zero_height_branch(self):
        """The '_' marker has all vertices at y=0, giving height_u=0."""
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0, 4.0, 5.0])
        h.add_points_above(3.0, marker="_")
        ax = h.ax
        assert ax is not None


class TestColorOverrideAlpha:
    def test_edgecolor_override_draws_opaque_edges(self):
        # An explicit edge color with no edge alpha must remain opaque.
        h = Histogram()
        h.add_dataset([0.0, 1.0, 1.0, 2.0], edgecolor="black", edgewidth=0.7)
        bar_edge_rgba = h.ax.patches[0].get_edgecolor()
        assert bar_edge_rgba == (0.0, 0.0, 0.0, 1.0)

    def test_explicit_edgealpha_still_wins(self):
        h = Histogram()
        h.add_dataset([0.0, 1.0, 1.0, 2.0], edgecolor="black", edgealpha=0.4, edgewidth=0.7)
        bar_edge_rgba = h.ax.patches[0].get_edgecolor()
        assert bar_edge_rgba == (0.0, 0.0, 0.0, 0.4)


@pytest.mark.parametrize(
    ("weights", "expected_heights"),
    [(None, [2.0, 1.0]), ([2.0, 3.0, 4.0], [5.0, 4.0])],
    ids=["counts", "weights"],
)
def test_bar_heights_match_histogram_counts_and_weights(weights, expected_heights):
    h = Histogram()
    h.set_bins([0.0, 1.0, 2.0])
    h.add_dataset([0.2, 0.7, 1.4], weights=weights)

    bars = [patch for patch in h.ax.patches if isinstance(patch, Rectangle)]
    assert [patch.get_height() for patch in bars] == expected_heights


class TestGroupedCenteredBinAlignment:
    def test_grouped_centered_group_tiles_around_bin_edge(self):
        # Regression: grouped bars were offset from the raw edges but drawn align="center",
        # shifting each group right by a quarter binwidth.
        h = Histogram()
        h.set_bins([0.0, 1.0, 2.0])
        h.add_dataset([0.5, 1.5], histtype="grouped", name="a")
        h.add_dataset([0.5, 1.5], histtype="grouped", name="b")
        h.center_bars()
        ax = h.ax

        from matplotlib.patches import Rectangle

        bars = sorted(
            (patch.get_x(), patch.get_width())
            for patch in ax.patches
            if isinstance(patch, Rectangle)
        )
        # Each bin's group spans [edge - 0.5, edge + 0.5], tiled by the two half-width bars.
        assert bars == pytest.approx([(-0.5, 0.5), (0.0, 0.5), (0.5, 0.5), (1.0, 0.5)])

    def test_points_above_center_on_displayed_bars_when_centered(self):
        h = Histogram()
        h.set_bins([0.0, 1.0, 2.0])
        h.add_dataset([0.5, 1.5], histtype="grouped", name="a")
        h.add_dataset([0.5, 1.5], histtype="grouped", name="b")
        h.center_bars()
        h.add_points_above(0.5, name="pt", centered_on_bin=True)
        ax = h.ax

        import numpy as np

        (point_line,) = ax.lines
        # Value 0.5 falls in bin [0, 1), whose bar group is visually centered on edge 0.
        assert np.asarray(point_line.get_xdata(), dtype=float) == pytest.approx([0.0])

    def test_points_use_displayed_bin_height_when_centered(self):
        h = Histogram()
        h.set_bins([0.0, 1.0, 2.0])
        h.add_dataset([0.1] * 5 + [1.1])
        h.center_bars()
        h.add_points_above(0.9)

        point_y = float(np.asarray(h.ax.lines[-1].get_ydata())[0])
        assert 1.0 < point_y < 5.0


class TestPointsAbovePlacement:
    """Regression tests: the point-space clearance must convert through a realized view.

    A stale transData on the first build collapsed the clearance to a near-zero epsilon,
    and stale data limits from a previous build's artists shifted markers on rebuilds.
    """

    def test_marker_positions_identical_across_rebuild(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 2.0, 3.0, 4.0])
        h.set_bins([1.0, 2.0, 3.0, 4.0])
        h.add_points_above([2.5, 2.5, 2.5], name="Plans")
        first_ys = np.asarray(h.ax.lines[-1].get_ydata(), dtype=float).copy()

        h.display_grid(False)  # dirty the plot so the next .ax access rebuilds
        second_ys = np.asarray(h.ax.lines[-1].get_ydata(), dtype=float)

        np.testing.assert_array_equal(first_ys, second_ys)

    def test_first_build_clearance_uses_realized_view(self):
        # A tall bar makes the y-range ~100; the stale default (0, 1) view produced a
        # clearance of ~0.01 data units, while the realized view gives well over 0.5.
        h = Histogram()
        h.add_dataset([1.5] * 100 + [2.5])
        h.set_bins([1.0, 2.0, 3.0])
        h.add_points_above(2.5)
        marker_y = float(np.asarray(h.ax.lines[-1].get_ydata(), dtype=float)[0])
        assert marker_y - 1.0 > 0.5


class TestStackedDensityNormalization:
    @staticmethod
    def _total_bar_area(ax):
        from matplotlib.patches import Rectangle

        return sum(
            bar.get_height() * bar.get_width() for bar in ax.patches if isinstance(bar, Rectangle)
        )

    def test_stacked_density_total_area_is_one(self):
        rng = np.random.default_rng(0)
        h = Histogram()
        h.as_density(True)
        h.add_dataset(rng.normal(0.0, 1.0, 500), histtype="stack")
        h.add_dataset(rng.normal(2.0, 1.0, 300), histtype="stack")
        assert self._total_bar_area(h.ax) == pytest.approx(1.0)

    def test_overlay_density_still_normalizes_per_series(self):
        rng = np.random.default_rng(1)
        h = Histogram()
        h.as_density(True)
        h.add_dataset(rng.normal(0.0, 1.0, 400))
        h.add_dataset(rng.normal(1.0, 1.0, 200))
        assert self._total_bar_area(h.ax) == pytest.approx(2.0)


# =================================
# == DENSITY ZERO-WEIGHT ERRORS  ==
# =================================


class TestDensityZeroWeights:
    """All-zero weights in density mode raise instead of rendering an all-NaN empty plot."""

    def test_plain_density_zero_weights_raises(self):
        h = Histogram()
        h.as_density(True)
        h.add_dataset([1.0, 2.0, 3.0], weights=[0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="sum to zero"):
            _ = h.ax

    def test_stacked_density_zero_weights_raises(self):
        h = Histogram()
        h.as_density(True)
        h.add_dataset([1.0, 2.0], weights=[0.0, 0.0], histtype="stack")
        h.add_dataset([2.0, 3.0], weights=[0.0, 0.0], histtype="stack")
        with pytest.raises(ValueError, match="sum to zero"):
            _ = h.ax

    def test_zero_weights_without_density_still_builds(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0, 3.0], weights=[0.0, 0.0, 0.0])
        assert h.ax is not None


# ================================
# == DATASET LEGEND HANDLE ORDER ==
# ================================


class TestDatasetLegendHandleOrder:
    """Legend handles order by histtype dict order (overlay, grouped, outline, stack),
    not by the order datasets were added."""

    def test_legend_handles_follow_histtype_order(self):
        h = Histogram()
        h.add_dataset([1.0, 2.0], histtype="stack", name="S")
        h.add_dataset(
            [1.0, 2.0],
            histtype="outline",
            name="L",
            facecolor="none",
            edgecolor="black",
            edgewidth=1.0,
        )
        h.add_dataset([1.0, 2.0], histtype="grouped", name="G")
        h.add_dataset([1.0, 2.0], histtype="overlay", name="O")
        labels = [handle.get_label() for handle in h._dataset_legend_handles()]
        assert labels == ["O", "G", "L", "S"]
