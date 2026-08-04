"""End-to-end coverage of gerrytools composing with raw matplotlib.

Once a user has an :class:`~matplotlib.axes.Axes` — either because they
bound a gerrytools plot to it or pulled one out via ``plot.ax`` — that axes
is a shared matplotlib surface. Most-recent-wins
per setting: whichever side (gerrytools or the user) touched a unit last
should be respected, and anything the user drew directly on the axes must
survive future gerrytools rebuilds.

These tests exercise the five concrete patterns we care about:

- subplot-grid embedding — the plot fills exactly its target cell;
- overlay on existing content (e.g. ``ax.imshow(backdrop)``) — the backdrop
  survives the gerrytools render;
- post-render customization — annotations added via ``ax.text(...)`` between
  renders survive the next rebuild;
- post-render axes-state mutation — ``ax.set_xlim(...)`` after ``.ax``
  survives the next rebuild;
- pre-render axes-state configuration — ``ax.set_xlim(...)`` *before*
  ``Histogram(ax=ax)`` survives the first render.

Both a data plot (``Histogram``) and a geometry plot (``GeoPlot``)
are exercised where the scenario is meaningful for both families.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from gerrytools.plotting.data.histogram import Histogram  # noqa: E402
from gerrytools.plotting.geometry.geoplot import GeoPlot  # noqa: E402

# ---------------------------------------------------------------------------
# Subplot grid embedding
# ---------------------------------------------------------------------------


class TestSubplotGridEmbedding:
    def test_histogram_renders_in_one_subplot_cell_only(self):
        fig, axes = plt.subplots(2, 2)
        hist = Histogram(ax=axes[0, 0])
        hist.add_dataset([1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0])
        hist.ax  # triggers render

        # Target cell has artists; sibling cells were untouched.
        assert len(axes[0, 0].patches) > 0
        for row, col in [(0, 1), (1, 0), (1, 1)]:
            assert len(axes[row, col].patches) == 0
            assert len(axes[row, col].lines) == 0

    def test_geoplot_renders_in_one_subplot_cell_only(self, testing_gdf):
        fig, axes = plt.subplots(2, 2)
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        plot.bind_to_ax(axes[0, 0])

        assert (len(axes[0, 0].patches) + len(axes[0, 0].lines) + len(axes[0, 0].collections)) > 0
        for row, col in [(0, 1), (1, 0), (1, 1)]:
            assert (
                len(axes[row, col].patches)
                + len(axes[row, col].lines)
                + len(axes[row, col].collections)
            ) == 0


# ---------------------------------------------------------------------------
# Overlay on existing content
# ---------------------------------------------------------------------------


class TestOverlayOnExistingContent:
    def test_imshow_backdrop_survives_histogram_render(self):
        fig, ax = plt.subplots()
        ax.imshow([[0.1, 0.2], [0.3, 0.4]])
        images_before = len(ax.images)
        hist = Histogram(ax=ax)
        hist.add_dataset([1.0, 2.0, 3.0])
        hist.ax
        assert len(ax.images) == images_before
        # And re-rendering preserves it too.
        hist.ax
        assert len(ax.images) == images_before

    def test_imshow_backdrop_survives_geoplot_render(self, testing_gdf):
        fig, ax = plt.subplots()
        ax.imshow([[0.1, 0.2], [0.3, 0.4]])
        images_before = len(ax.images)
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        plot.bind_to_ax(ax)
        assert len(ax.images) == images_before


# ---------------------------------------------------------------------------
# Post-render customization (external annotations)
# ---------------------------------------------------------------------------


class TestPostRenderCustomization:
    def test_external_text_survives_subsequent_histogram_rebuild(self):
        hist = Histogram()
        hist.add_dataset([1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0])
        ax = hist.ax
        ax.text(2.0, 0.5, "Note", color="red")
        hist.ax  # second render
        matching = [t for t in ax.texts if t.get_text() == "Note"]
        assert len(matching) == 1

    def test_external_text_survives_subsequent_geoplot_rebuild(self, testing_gdf):
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        ax = plot.ax
        ax.text(0.5, 0.5, "Note", transform=ax.transAxes)
        plot.ax  # second render
        matching = [t for t in ax.texts if t.get_text() == "Note"]
        assert len(matching) == 1


# ---------------------------------------------------------------------------
# Post-render axes-state mutation
# ---------------------------------------------------------------------------


class TestPostRenderAxesStateMutation:
    def test_external_xlim_after_render_survives_rebuild_histogram(self):
        hist = Histogram()
        hist.add_dataset([1.0, 2.0, 3.0, 4.0, 5.0])
        ax = hist.ax
        ax.set_xlim(0.0, 100.0)
        hist.ax
        assert hist._ax.get_xlim() == (0.0, 100.0)

    def test_external_ylim_after_render_survives_rebuild_histogram(self):
        hist = Histogram()
        hist.add_dataset([1.0, 2.0, 3.0, 4.0, 5.0])
        ax = hist.ax
        ax.set_ylim(0.0, 999.0)
        hist.ax
        assert hist._ax.get_ylim() == (0.0, 999.0)

    def test_external_xlim_after_render_survives_rebuild_geoplot(self, testing_gdf):
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        ax = plot.ax
        ax.set_xlim(-1000.0, 1000.0)
        plot.ax
        assert plot._ax.get_xlim() == (-1000.0, 1000.0)


# ---------------------------------------------------------------------------
# Pre-render axes-state configuration
# ---------------------------------------------------------------------------


class TestPreConfiguredAxes:
    def test_pre_set_xlim_preserved_after_histogram_first_render(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0.0, 50.0)
        hist = Histogram(ax=ax)
        hist.add_dataset([1.0, 2.0, 3.0])
        hist.ax
        assert ax.get_xlim() == (0.0, 50.0)

    def test_pre_set_title_preserved_after_histogram_first_render(self):
        fig, ax = plt.subplots()
        ax.set_title("user title")
        hist = Histogram(ax=ax)  # title=None default = "no opinion"
        hist.add_dataset([1.0, 2.0, 3.0])
        hist.ax
        assert ax.get_title() == "user title"

    def test_pre_set_ylim_preserved_after_histogram_first_render(self):
        fig, ax = plt.subplots()
        ax.set_ylim(0.0, 50.0)
        hist = Histogram(ax=ax)
        hist.add_dataset([1.0, 2.0, 3.0])
        hist.ax
        assert ax.get_ylim() == (0.0, 50.0)

    @pytest.mark.parametrize(
        ("axis", "positions"),
        [("x", [0.0, 1.5, 3.0]), ("y", [0.0, 2.0, 4.0])],
    )
    def test_pre_set_ticks_preserved_after_histogram_first_render(self, axis, positions):
        fig, ax = plt.subplots()
        getattr(ax, f"set_{axis}ticks")(positions)
        hist = Histogram(ax=ax)
        hist.add_dataset([1.0, 2.0, 3.0])
        hist.ax
        assert getattr(ax, f"get_{axis}ticks")().tolist() == positions

    def test_pre_set_xlim_preserved_after_geoplot_first_render(self, testing_gdf):
        fig, ax = plt.subplots()
        ax.set_xlim(0.0, 50.0)
        plot = GeoPlot(testing_gdf)
        plot.add_outline_layer()
        plot.bind_to_ax(ax)
        assert ax.get_xlim() == (0.0, 50.0)


# ---------------------------------------------------------------------------
# Chained: most-recent-wins across multiple touches
# ---------------------------------------------------------------------------


class TestChainedTouches:
    def test_external_xlim_then_explicit_gerrytools_setter_wins(self):
        """After an external ``ax.set_xlim`` yields the unit to external
        state, a subsequent gerrytools ``set_xlim`` reclaims and wins."""
        hist = Histogram()
        hist.add_dataset([1.0, 2.0, 3.0, 4.0, 5.0])
        ax = hist.ax
        ax.set_xlim(0.0, 100.0)
        hist.ax
        assert hist._ax.get_xlim() == (0.0, 100.0)
        hist.set_xlim(-10.0, 10.0)
        hist.ax
        assert hist._ax.get_xlim() == (-10.0, 10.0)

    def test_imshow_backdrop_and_post_render_annotation_both_survive(self):
        """Backdrop imshow AND a post-render annotation should both
        survive the next rebuild."""
        fig, ax = plt.subplots()
        ax.imshow([[0.1, 0.2], [0.3, 0.4]])
        hist = Histogram(ax=ax)
        hist.add_dataset([1.0, 2.0, 3.0])
        hist.ax  # first render
        ax.text(1.0, 0.0, "Annotation", color="red")
        hist.ax  # second render
        assert len(ax.images) == 1
        assert any(t.get_text() == "Annotation" for t in ax.texts)


# ---------------------------------------------------------------------------
# matplotlib environment canary: annotation handles must stay removable
# ---------------------------------------------------------------------------


class TestAnnotationHandlesAreRemovable:
    """Tripwire for the artist registry's core assumption.

    The registry replaces ``ax.clear()`` by calling ``.remove()`` on each
    artist gerrytools created during a render. That only works while
    matplotlib's annotation APIs keep returning a live, removable handle.
    If a future matplotlib version changes any of these to return ``None``
    or an object whose ``.remove()`` is a no-op, the registry would
    silently leak artists across rebuilds (the registry swallows removal
    errors by design, so the failure would be invisible in normal use).

    These run once and fail loudly the moment that contract breaks, so a
    matplotlib upgrade surfaces the regression here rather than as drifting
    artist counts in unrelated plot tests.
    """

    def test_axvline_returns_removable_handle(self):
        self._assert_removable(lambda ax: ax.axvline(0.5))

    def test_axhline_returns_removable_handle(self):
        self._assert_removable(lambda ax: ax.axhline(0.5))

    def test_axvspan_returns_removable_handle(self):
        self._assert_removable(lambda ax: ax.axvspan(0.1, 0.2))

    def test_axhspan_returns_removable_handle(self):
        self._assert_removable(lambda ax: ax.axhspan(0.1, 0.2))

    def test_annotate_returns_removable_handle(self):
        self._assert_removable(lambda ax: ax.annotate("note", (0.5, 0.5)))

    @staticmethod
    def _assert_removable(draw):
        """Draw via ``draw(ax)`` and assert the handle is live and removable.

        Membership is checked against ``ax.get_children()`` rather than a
        specific container (``ax.lines`` / ``ax.patches`` / ``ax.texts``) so
        the canary does not care which list a given matplotlib version files
        the artist under — only that the returned handle is attached and that
        ``.remove()`` actually detaches it.
        """
        fig, ax = plt.subplots()
        try:
            handle = draw(ax)
            assert handle is not None, "matplotlib returned no handle"
            assert hasattr(handle, "remove"), f"{type(handle)!r} is not removable"
            assert handle in ax.get_children(), "handle was not attached to the axes"
            handle.remove()
            assert handle not in ax.get_children(), "handle survived .remove()"
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Artist registry: removal failures must not abort the rebuild sweep
# ---------------------------------------------------------------------------


class TestArtistRegistryRemovalTolerance:
    def test_remove_all_survives_non_removable_artist_and_clears_tracking(self):
        """A tracked artist whose ``.remove()`` raises (a bare ``Artist`` has no remove
        method) is skipped: later artists still detach and the tracking list empties."""
        from matplotlib.artist import Artist

        from gerrytools.plotting._artist_registry import _ArtistRegistry

        fig, ax = plt.subplots()
        try:
            non_removable = Artist()  # .remove() raises NotImplementedError
            removable_line = ax.axvline(0.5)
            registry = _ArtistRegistry()
            registry.track([non_removable, removable_line])

            registry.remove_all()

            assert removable_line not in ax.get_children()
            assert registry._tracked == []
        finally:
            plt.close(fig)

    def test_bar_containers_do_not_accumulate_across_rebuilds(self):
        from gerrytools.plotting.data.barplot import BarPlot

        plot = BarPlot(legend=False)
        plot.add_dataset({"A": 1.0})
        for title in ("one", "two", "three"):
            plot.title = title
            plot.ax

        assert len(plot.ax.containers) == 1


# ---------------------------------------------------------------------------
# Grid is tri-state: no opinion by default, explicit True/False still applies
# ---------------------------------------------------------------------------


class TestExternalGridSurvives:
    @staticmethod
    def _grid_visible(ax) -> bool:
        gridlines = ax.xaxis.get_gridlines()
        return bool(gridlines) and gridlines[0].get_visible()

    def test_histogram_leaves_external_grid_alone(self):
        fig, ax = plt.subplots()
        ax.grid(True)
        hist = Histogram(ax=ax)
        hist.add_dataset([1.0, 2.0, 3.0, 3.0, 4.0])
        hist.ax
        assert self._grid_visible(ax)
        plt.close(fig)

    def test_histogram_explicit_grid_false_disables(self):
        fig, ax = plt.subplots()
        ax.grid(True)
        hist = Histogram(ax=ax)
        hist.display_grid(False)
        hist.add_dataset([1.0, 2.0, 3.0, 3.0, 4.0])
        hist.ax
        assert not self._grid_visible(ax)
        plt.close(fig)

    def test_histogram_explicit_grid_true_enables(self):
        hist = Histogram()
        hist.display_grid(True)
        hist.add_dataset([1.0, 2.0, 3.0])
        assert self._grid_visible(hist.ax)

    def test_sealevel_leaves_external_grid_alone(self):
        from gerrytools.plotting.data.sealevel import SeaLevelPlot

        fig, ax = plt.subplots()
        ax.grid(True)
        plot = SeaLevelPlot(ax=ax)
        plot.add_dataset({"A": 0.5, "B": 0.7})
        plot.ax
        assert self._grid_visible(ax)
        plt.close(fig)

    def test_sealevel_explicit_grid_false_disables(self):
        from gerrytools.plotting.data.sealevel import SeaLevelPlot

        fig, ax = plt.subplots()
        ax.grid(True)
        plot = SeaLevelPlot(ax=ax)
        plot.display_grid(False)
        plot.add_dataset({"A": 0.5, "B": 0.7})
        plot.ax
        assert not self._grid_visible(ax)
        plt.close(fig)
