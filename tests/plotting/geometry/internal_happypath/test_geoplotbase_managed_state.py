"""Regression tests for the artist-registry / managed-axes-state design in geometry plots.

Geometry-side coverage of the contract gerrytools plots maintain on a
shared matplotlib axes:

- artist counts stay flat across N rebuilds (no leak);
- external matplotlib content (text, imshow) on a shared axes survives
  GeoPlotBase/GeoPlot rebuilds;
- ``_figure_is_shared`` blocks ``subplots_adjust`` mutation when the user
  bound their plot to an axes;
- ``show_axis`` is a managed unit (most-recent-wins between gerrytools and
  external ``ax.set_axis_on()``/``set_axis_off()``);
- ``set_xlim``/``set_ylim`` reclaim per the same managed-unit contract as
  the data-plot side.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from gerrytools.plotting.geometry.geoplot import GeoPlot


def _total_artist_count(ax) -> int:
    return len(ax.patches) + len(ax.lines) + len(ax.collections) + len(ax.texts) + len(ax.images)


# ---------------------------------------------------------------------------
# No-leak guardrails
# ---------------------------------------------------------------------------


class TestNoLeakAcrossRebuilds:
    def test_geoplot_artist_counts_flat_across_rebuilds(self, testing_gdf):
        # GeoPlotBase's _build_plot is abstract; use GeoPlot as the
        # concrete proxy since GeoPlotBase is also abstract for instantiation.
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        counts = [_total_artist_count(plot.ax) for _ in range(4)]
        assert counts[0] > 0
        assert len(set(counts)) == 1, f"artist counts drift: {counts}"


# ---------------------------------------------------------------------------
# External content preservation on shared axes
# ---------------------------------------------------------------------------


class TestExternalContentSurvives:
    def test_external_text_survives_geoplot_rebuild(self, testing_gdf):
        _, ax = plt.subplots()
        ax.text(0.5, 0.5, "external", transform=ax.transAxes)
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        plot.bind_to_ax(ax)
        plot.ax
        plot.ax  # second rebuild
        matching = [t for t in ax.texts if t.get_text() == "external"]
        assert len(matching) == 1

    def test_external_data_contributes_to_autoscale(self, testing_gdf):
        _, ax = plt.subplots()
        ax.plot([1_000_000, 1_000_001], [1_000_000, 1_000_001])
        ax.scatter([2_000_000], [2_000_000])
        plot = GeoPlot(testing_gdf, default_outline=False)
        plot.add_outline_layer()
        plot.bind_to_ax(ax)

        xlim = plot.ax.get_xlim()

        assert xlim[0] <= testing_gdf.total_bounds[0]
        assert xlim[1] >= 2_000_000


# ---------------------------------------------------------------------------
# _figure_is_shared
# ---------------------------------------------------------------------------


class TestFigureIsShared:
    def test_figure_is_shared_flag_set_after_binding(self, testing_gdf):
        _, ax = plt.subplots()
        plot = GeoPlot(testing_gdf)
        plot.bind_to_ax(ax)
        assert plot._figure_is_shared is True

    def test_figure_is_shared_flag_unset_when_gerrytools_creates_figure(self, testing_gdf):
        plot = GeoPlot(testing_gdf)
        assert plot._figure_is_shared is False

    def test_subplots_adjust_skipped_on_shared_figure(self, testing_gdf):
        """When the user owns the figure, gerrytools must not call
        ``subplots_adjust`` — that would shift other axes the user owns.
        """
        fig, ax = plt.subplots()
        # Pre-set right=0.5 — a value gerrytools would normally reset to 0.98.
        fig.subplots_adjust(right=0.5)
        before_right = fig.subplotpars.right
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        plot.bind_to_ax(ax)
        # subplots_adjust must NOT have run, so the right margin is unchanged.
        assert fig.subplotpars.right == before_right

    def test_subplots_adjust_runs_on_owned_figure(self, testing_gdf):
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        plot.ax
        # gerrytools-owned figure: subplots_adjust ran, so right is the
        # default-reset value 0.98 (no colorbar requests in this case).
        assert plot.fig.subplotpars.right == 0.98


# ---------------------------------------------------------------------------
# Managed title
# ---------------------------------------------------------------------------


class TestManagedTitle:
    def test_constructor_title_and_style_survive_binding(self, testing_gdf):
        plot = GeoPlot(testing_gdf, title="Georgia")
        plot.set_title_style(fontsize=18, loc="left")
        _, ax = plt.subplots()

        plot.bind_to_ax(ax)

        assert ax.get_title(loc="left") == "Georgia"
        assert getattr(ax, "_left_title").get_fontsize() == 18

    def test_omitted_title_preserves_external_title(self, testing_gdf):
        _, ax = plt.subplots()
        ax.set_title("External")
        plot = GeoPlot(testing_gdf)

        plot.bind_to_ax(ax)

        assert ax.get_title() == "External"

    def test_title_property_reclaims_after_external_change(self, testing_gdf):
        plot = GeoPlot(testing_gdf, title="First")
        ax = plot.ax
        ax.set_title("External")
        plot.title = "Second"

        assert plot.ax.get_title() == "Second"


# ---------------------------------------------------------------------------
# show_axis managed unit
# ---------------------------------------------------------------------------


class TestShowAxisManagedUnit:
    def test_constructor_show_axis_true_renders_axis_on(self, testing_gdf):
        plot = GeoPlot(testing_gdf, show_axis=True)
        plot.add_outline_layer()
        plot.ax
        assert plot._ax.axison

    def test_constructor_default_renders_axis_off(self, testing_gdf):
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        plot.ax
        assert not plot._ax.axison

    def test_show_axis_setter_most_recent_wins(self, testing_gdf):
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        plot.ax
        assert not plot._ax.axison
        # Gerrytools setter wins.
        plot.show_axis = True
        plot.ax
        assert plot._ax.axison
        # External setter wins over prior gerrytools state.
        plot._ax.set_axis_off()
        plot.ax
        assert not plot._ax.axison
        # Gerrytools setter wins again.
        plot.show_axis = True
        plot.ax
        assert plot._ax.axison


# ---------------------------------------------------------------------------
# Limits managed unit
# ---------------------------------------------------------------------------


class TestLimitsManagedUnit:
    def test_set_xlim_then_external_set_xlim_external_wins(self, testing_gdf):
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        plot.set_xlim(-100.0, 100.0)
        ax = plot.ax
        assert ax.get_xlim() == (-100.0, 100.0)
        ax.set_xlim(0.0, 50.0)
        plot.ax
        assert plot._ax.get_xlim() == (0.0, 50.0)

    def test_explicit_xlim_survives_bind_to_ax(self, testing_gdf):
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        plot.set_xlim(-100.0, 100.0)
        plot.ax
        _, ax2 = plt.subplots()
        plot.bind_to_ax(ax2)
        plot.ax
        assert plot._ax is ax2
        assert plot._ax.get_xlim() == (-100.0, 100.0)
