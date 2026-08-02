"""Tests for axes-backed plot construction and ``bind_to_ax``.

Data plots accept a Matplotlib ``Axes`` at construction time. All axes-backed plots can rebind an
existing plot later; geometry plots use only that explicit binding path.
"""

from __future__ import annotations

import inspect
from typing import cast

import matplotlib

matplotlib.use("Agg")

import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import SubFigure  # noqa: E402

from gerrytools.plotting import DotDensityPlot, GeoPlot, Histogram  # noqa: E402
from gerrytools.plotting.geometry.geoplotbase import GeoPlotBase  # noqa: E402
from tests.plotting._typing_utils import as_any

# ----------------------------------------------------------------------
# Constructor accepts ax=
# ----------------------------------------------------------------------


class TestAxConstructorParameter:
    def test_user_provided_ax_is_returned_by_plot_ax(self):
        fig, user_ax = plt.subplots(figsize=(4, 4))
        plot = Histogram(ax=user_ax)
        plot.add_dataset([1.0, 2.0, 3.0, 2.0, 1.0])
        assert plot.ax is user_ax

    def test_user_provided_fig_is_used(self):
        fig, user_ax = plt.subplots(figsize=(4, 4))
        plot = Histogram(ax=user_ax)
        assert plot.fig is fig

    def test_subfigure_axes_use_root_figure_for_save(self, tmp_path):
        fig = plt.figure()
        subfigure = cast("SubFigure", fig.subfigures(1, 1))
        user_ax = subfigure.subplots()
        plot = Histogram(ax=user_ax)
        plot.add_dataset([1.0, 2.0, 3.0])

        output = tmp_path / "subfigure.png"
        plot.save(str(output))

        assert plot.fig is fig
        assert output.exists()

    def test_no_ax_creates_fresh_figure(self):
        plot = Histogram()
        plot.add_dataset([1.0, 2.0, 3.0])
        assert plot.fig is not None
        assert plot.ax is not None

    def test_figure_size_and_ax_together_warns(self):
        fig, user_ax = plt.subplots()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Histogram(figure_size=(8, 8), ax=user_ax)
        assert any("ignored when ax is provided" in str(w.message) for w in caught)

    def test_dpi_and_ax_together_warns(self):
        fig, user_ax = plt.subplots()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Histogram(dpi=72, ax=user_ax)
        assert any("ignored when ax is provided" in str(w.message) for w in caught)

    def test_ax_alone_does_not_warn(self):
        fig, user_ax = plt.subplots()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Histogram(ax=user_ax)
        assert not any("ignored when ax is provided" in str(w.message) for w in caught)


# ----------------------------------------------------------------------
# Deferred axis updates
# ----------------------------------------------------------------------


class TestDeferredAxisUpdates:
    def test_repeated_ax_access_builds_only_once_until_plot_changes(self, monkeypatch):
        plot = Histogram()
        plot.add_dataset([1.0, 2.0, 3.0])
        plot.ax

        build_count = 0
        original_build = plot._build_and_apply_settings

        def counted_build():
            nonlocal build_count
            build_count += 1
            return original_build()

        monkeypatch.setattr(plot, "_build_and_apply_settings", counted_build)

        plot.ax
        assert build_count == 0

        plot.add_dataset([10.0, 20.0, 30.0])
        plot.ax
        plot.ax
        assert build_count == 1

    def test_external_axis_change_survives_next_deferred_rebuild(self):
        plot = Histogram()
        plot.add_dataset([1.0, 2.0, 3.0])
        ax = plot.ax

        ax.set_title("from matplotlib")
        assert plot.ax.get_title() == "from matplotlib"

        plot.add_dataset([10.0, 20.0, 30.0])
        assert plot.ax.get_title() == "from matplotlib"

    def test_failed_build_remains_pending(self, monkeypatch):
        plot = Histogram()
        plot.add_dataset([1.0, 2.0, 3.0])

        def fail_build():
            raise RuntimeError("build failed")

        monkeypatch.setattr(plot, "_build_and_apply_settings", fail_build)

        with pytest.raises(RuntimeError, match="build failed"):
            plot.ax

        assert plot._axis_needs_update

    def test_direct_layout_property_change_schedules_update(self):
        from gerrytools.plotting import BoxPlot

        plot = BoxPlot()
        plot.add_dataset({"A": [1.0, 2.0, 3.0]})
        plot.ax
        assert not plot._axis_needs_update

        plot.group_width = 0.5
        assert plot._axis_needs_update
        plot.ax
        assert not plot._axis_needs_update

    def test_geoplot_label_positions_rebuild_after_rebind(self, monkeypatch):
        plot = GeoPlot(_simple_gdf(), default_outline=False)
        plot.get_label_positions()

        build_count = 0
        original_build = plot._build_and_apply_settings

        def counted_build():
            nonlocal build_count
            build_count += 1
            return original_build()

        monkeypatch.setattr(plot, "_build_and_apply_settings", counted_build)
        _, new_ax = plt.subplots()
        plot.bind_to_ax(new_ax)
        plot.get_label_positions()

        assert build_count == 1


# ----------------------------------------------------------------------
# bind_to_ax retargets
# ----------------------------------------------------------------------


class TestBindToAx:
    def test_bind_to_ax_immediately_builds_pending_geoplot(self):
        plot = GeoPlot(_simple_gdf())
        plot.add_districting_plan_layer("name")
        _, new_ax = plt.subplots()

        plot.bind_to_ax(new_ax)

        assert len(new_ax.collections) == 2
        assert not plot._axis_needs_update

    def test_bind_to_ax_retargets_to_new_axes(self):
        plot = Histogram()
        plot.add_dataset([1.0, 2.0, 3.0])
        _ = plot.ax  # build onto plot's own fig

        _, new_ax = plt.subplots()
        plot.bind_to_ax(new_ax)
        assert plot.ax is new_ax

    def test_bind_to_ax_none_reverts_to_fresh_figure(self):
        _, user_ax = plt.subplots()
        plot = Histogram(ax=user_ax)
        assert plot._ax is user_ax
        plot.bind_to_ax(None)
        # After unbinding, _ax points to a fresh axes
        assert plot._ax is not user_ax

    def test_rebind_preserves_added_data(self):
        plot = Histogram()
        plot.add_dataset([1.0, 2.0, 3.0, 4.0], name="series_a")
        _, new_ax = plt.subplots()
        plot.bind_to_ax(new_ax)
        # The data should still be there for the new render
        assert len(plot._hist_data_dict["overlay"]) == 1
        assert plot._hist_data_dict["overlay"][0].name == "series_a"

    def test_bind_to_ax_none_preserves_construction_figure_size_and_dpi(self):
        plot = Histogram(figure_size=(4.0, 3.0), dpi=150)
        plot.bind_to_ax(None)
        assert tuple(plot.fig.get_size_inches()) == (4.0, 3.0)
        assert plot.fig.dpi == 150

    def test_bind_to_ax_none_matches_default_construction_geometry(self):
        plot = Histogram()
        plot.bind_to_ax(None)
        assert tuple(plot.fig.get_size_inches()) == (10.0, 6.0)
        assert plot.fig.dpi == 300

    def test_bind_to_ax_none_uses_defaults_when_constructed_with_user_ax(self):
        # A plot constructed onto a user axes has no construction geometry of
        # its own; unbinding falls back to the standard defaults rather than
        # inheriting the user figure's geometry.
        _, user_ax = plt.subplots(figsize=(2.0, 2.0), dpi=72)
        plot = Histogram(ax=user_ax)
        plot.bind_to_ax(None)
        assert tuple(plot.fig.get_size_inches()) == (10.0, 6.0)
        assert plot.fig.dpi == 300


# ----------------------------------------------------------------------
# GeoPlot binding
# ----------------------------------------------------------------------


class TestGeoPlotBinding:
    @pytest.mark.parametrize("plot_type", [GeoPlotBase, GeoPlot, DotDensityPlot])
    def test_geometry_constructor_has_no_ax_parameter(self, plot_type):
        assert "ax" not in inspect.signature(plot_type).parameters

    def test_geoplot_bind_to_ax(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"name": ["A"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
            crs="EPSG:4326",
        )
        plot = GeoPlot(gdf)
        _, new_ax = plt.subplots()
        plot.bind_to_ax(new_ax)
        assert plot._ax is new_ax

    def test_geoplot_bind_to_ax_none_preserves_construction_dpi_and_remains_lazy(self):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"name": ["A"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
            crs="EPSG:4326",
        )
        plot = GeoPlot(gdf, dpi=150)
        plot.add_districting_plan_layer("name")
        plot.bind_to_ax(None)

        assert plot.fig.dpi == 150
        assert plot._axis_needs_update
        assert not plot._ax.collections

        assert len(plot.ax.collections) == 2
        assert not plot._axis_needs_update


# ----------------------------------------------------------------------
# Figure lifecycle: self-created figures close on garbage collection
# ----------------------------------------------------------------------


def _simple_gdf():
    import geopandas as gpd
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame(
        {"name": ["A"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]},
        crs="EPSG:4326",
    )


class TestFigureLifecycle:
    """Self-created figures must not accumulate in pyplot's figure manager.

    ``plt.subplots()`` registers every figure with pyplot's global manager, which holds a strong
    reference until ``plt.close``. Both plot base classes register a ``weakref.finalize`` so a
    plot going out of scope closes its own figure; user-supplied figures are never touched.
    """

    def test_histogram_figures_close_when_plots_collected(self):
        import gc

        plt.close("all")
        plot: Histogram | None = None
        for _ in range(5):
            plot = Histogram()
            plot.add_dataset([1.0, 2.0, 3.0])
        assert plot is not None
        del plot
        gc.collect()
        assert plt.get_fignums() == []

    def test_geoplot_figures_close_when_plots_collected(self):
        import gc

        plt.close("all")
        gdf = _simple_gdf()
        plot: GeoPlot | None = None
        for _ in range(5):
            plot = GeoPlot(gdf)
        assert plot is not None
        del plot
        gc.collect()
        assert plt.get_fignums() == []

    def test_user_axes_figure_survives_plot_collection(self):
        import gc

        plt.close("all")
        fig, user_ax = plt.subplots()
        plot = Histogram(ax=user_ax)
        del plot
        gc.collect()
        assert plt.get_fignums() == [fig.number]

        geo_plot = GeoPlot(_simple_gdf())
        geo_plot.bind_to_ax(user_ax)
        del geo_plot
        gc.collect()
        assert plt.get_fignums() == [fig.number]

    def test_rebind_away_from_owned_figure_closes_it(self):
        plt.close("all")
        plot = Histogram()
        plot.add_dataset([1.0, 2.0, 3.0])
        _ = plot.ax
        owned_fig_number = plot.fig.number
        _, user_ax = plt.subplots()
        plot.bind_to_ax(user_ax)
        assert owned_fig_number not in plt.get_fignums()

    def test_rebind_within_own_owned_figure_keeps_ownership(self):
        # Regression: rebinding to an axes on the plot's own owned figure used to detach
        # the finalizer and mark the figure shared, leaking it in pyplot's manager.
        import gc

        plt.close("all")
        plot = Histogram()
        plot.add_dataset([1.0, 2.0, 3.0])
        plot.bind_to_ax(plot.ax)
        del plot
        gc.collect()
        assert plt.get_fignums() == []

    def test_rebinding_to_current_axes_does_not_duplicate_artists(self):
        _, ax = plt.subplots()
        plot = Histogram(ax=ax)
        plot.add_dataset([1.0, 2.0, 3.0])
        expected = len(plot.ax.patches)

        plot.bind_to_ax(ax)
        assert len(plot.ax.patches) == expected
        plot.bind_to_ax(ax)
        assert len(plot.ax.patches) == expected

    def test_rebinding_back_to_previous_axes_replaces_managed_artists(self):
        _, ax_a = plt.subplots()
        _, ax_b = plt.subplots()
        plot = Histogram(ax=ax_a)
        plot.add_dataset([1.0, 2.0, 3.0])
        expected = len(plot.ax.patches)

        for target in (ax_b, ax_a, ax_b, ax_a):
            plot.bind_to_ax(target)
            assert len(target.patches) == expected

    def test_rebinding_to_current_axes_keeps_managed_limits_reactive(self):
        _, ax = plt.subplots()
        plot = Histogram(ax=ax)
        plot.add_dataset([0.0, 1.0])
        plot.ax

        plot.bind_to_ax(ax)
        plot.add_dataset([100.0, 101.0])

        assert plot.ax.get_xlim()[1] > 100.0

    def test_rebind_from_external_ax_leaves_figure_alone(self):
        plt.close("all")
        fig_a, ax_a = plt.subplots()
        fig_b, ax_b = plt.subplots()
        plot = Histogram(ax=ax_a)
        plot.add_dataset([1.0, 2.0, 3.0])
        plot.bind_to_ax(ax_b)
        assert fig_a.number in plt.get_fignums()

    def test_rebind_owned_to_none_closes_old_owned_figure(self):
        from matplotlib._pylab_helpers import Gcf

        plt.close("all")
        plot = Histogram()
        plot.add_dataset([1.0, 2.0, 3.0])
        old_fig = plot.fig
        plot.bind_to_ax(None)
        # The fresh figure may reuse the old figure *number*, so check identity.
        registered = [manager.canvas.figure for manager in Gcf.get_all_fig_managers()]
        assert old_fig not in registered
        assert plot.fig in registered
        assert plot.fig is not old_fig

    def test_geoplot_rebind_to_user_ax_detaches_finalizer(self):
        import gc

        plt.close("all")
        plot = GeoPlot(_simple_gdf())
        fig, user_ax = plt.subplots()
        plot.bind_to_ax(user_ax)
        del plot
        gc.collect()
        assert fig.number in plt.get_fignums()


# ----------------------------------------------------------------------
# Renamed methods are reachable
# ----------------------------------------------------------------------


class TestRenamedTickMethods:
    def test_old_tick_update_and_clear_spellings_are_gone(self):
        plot = Histogram()
        for old_name in (
            "update_xtick_values",
            "update_xtick_labels",
            "update_ytick_labels",
            "clear_xtick_labels",
            "clear_ytick_labels",
            "clear_xticks",
            "clear_yticks",
        ):
            assert not hasattr(plot, old_name)

    def test_set_xlimits_alias_is_gone(self):
        plot = Histogram()
        assert not hasattr(plot, "set_xlimits")


# ----------------------------------------------------------------------
# Enable/disable boolean setters (Theme 5)
# ----------------------------------------------------------------------


class TestEnableDisableSetters:
    def test_histogram_enable_grid(self):
        plot = Histogram()
        # Tri-state default: no opinion until display_grid is called.
        assert plot.grid is None
        plot.display_grid(True)
        assert plot.grid is True
        plot.display_grid(False)
        assert plot.grid is False

    def test_histogram_suppress_warnings(self):
        plot = Histogram()
        assert plot._show_warnings is True
        plot.display_warnings(False)
        assert plot._show_warnings is False
        plot.display_warnings(True)
        assert plot._show_warnings is True


# ----------------------------------------------------------------------
# PaintballPlot reshape (Theme 6)
# ----------------------------------------------------------------------


class TestPaintBallReshape:
    def test_empty_constructor_works(self):
        from gerrytools.plotting import PaintballPlot

        plot = PaintballPlot()
        assert plot._voteshare_data == []
        assert plot._seatshare_data == []

    def test_add_voteshare_seatshare_data_is_canonical(self):
        from gerrytools.plotting import PaintballPlot

        plot = PaintballPlot()
        plot.add_seats_votes_data([0.4, 0.5, 0.6], [0.3, 0.5, 0.7])
        assert plot._voteshare_data == [0.4, 0.5, 0.6]

    def test_no_default_guide_lines(self):
        from gerrytools.plotting import PaintballPlot

        plot = PaintballPlot()
        plot.add_seats_votes_data([0.5], [0.5])
        assert "Efficiency Gap" not in plot._named_lines
        assert "Proportionality" not in plot._named_lines

    def test_add_efficiency_gap_line(self):
        from gerrytools.plotting import PaintballPlot

        plot = PaintballPlot()
        plot.add_seats_votes_data([0.5], [0.5])
        plot.add_efficiency_gap_line()
        assert "Efficiency Gap" in plot._named_lines
        assert plot._named_lines["Efficiency Gap"].lines[0].slope == 2.0

    def test_add_proportionality_line(self):
        from gerrytools.plotting import PaintballPlot

        plot = PaintballPlot()
        plot.add_seats_votes_data([0.5], [0.5])
        plot.add_proportionality_line()
        assert "Proportionality" in plot._named_lines
        assert plot._named_lines["Proportionality"].lines[0].slope == 1.0

    def test_constructor_rejects_legacy_voteshare_kwarg(self):
        from gerrytools.plotting import PaintballPlot

        with pytest.raises(TypeError):
            PaintballPlot(**as_any({"vote_share_data": [0.5], "seats_data": [0.5]}))
