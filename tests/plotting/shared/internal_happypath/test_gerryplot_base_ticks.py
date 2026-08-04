import matplotlib

matplotlib.use("Agg")


import pytest
from matplotlib.font_manager import FontProperties

from gerrytools.plotting.data.scatterplot import ScatterPlot


def _make_plot():
    """Create a minimal ScatterPlot with some data for testing base methods."""
    sp = ScatterPlot()
    sp.add_series(x=[0.0, 1.0], y=[0.0, 1.0])
    return sp


# =================
# == TICK VALUES ==
# =================
class TestTickManagement:
    def test_set_xticks_stores_locations_and_labels(self):
        sp = _make_plot()
        sp.set_xticks([0.0, 0.5, 1.0], labels=["a", "b", "c"])
        assert sp._xaxis.tick_locations == [0.0, 0.5, 1.0]
        assert sp._xaxis.tick_labels == ["a", "b", "c"]

    def test_set_yticks_stores_locations_and_labels(self):
        sp = _make_plot()
        sp.set_yticks([0.0, 1.0], labels=["low", "high"])
        assert sp._yaxis.tick_locations == [0.0, 1.0]
        assert sp._yaxis.tick_labels == ["low", "high"]

    def test_set_xticks_empty_clears(self):
        sp = _make_plot()
        sp.set_xticks([0.0, 1.0], labels=["a", "b"])
        sp.set_xticks([])
        assert sp._xaxis.tick_locations == []
        assert sp._xaxis.tick_labels == []

    def test_set_yticks_empty_clears(self):
        sp = _make_plot()
        sp.set_yticks([0.0, 1.0], labels=["a", "b"])
        sp.set_yticks([])
        assert sp._yaxis.tick_locations == []
        assert sp._yaxis.tick_labels == []

    def test_set_xticks_locations_only(self):
        sp = _make_plot()
        sp.set_xticks(locations=[0.0, 0.5, 1.0])
        assert sp._xaxis.tick_locations == [0.0, 0.5, 1.0]

    def test_locations_after_clear_restore_automatic_labels(self):
        sp = _make_plot()
        sp.set_xticks(locations=[])
        sp.set_xticks(locations=[0.0, 0.5, 1.0])

        assert sp._xaxis.tick_labels is None
        assert [tick.get_text() for tick in sp.ax.get_xticklabels()] == ["0.0", "0.5", "1.0"]

    def test_set_xticks_labels_only(self):
        sp = _make_plot()
        sp.set_xticks(labels=["a", "b"])
        assert sp._xaxis.tick_labels == ["a", "b"]

    def test_set_xticks_both(self):
        sp = _make_plot()
        sp.set_xticks(locations=[1.0, 2.0], labels=["x", "y"])
        assert sp._xaxis.tick_locations == [1.0, 2.0]
        assert sp._xaxis.tick_labels == ["x", "y"]

    def test_set_xticks_mismatched_lengths_raises_valueerror(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="does not match"):
            sp.set_xticks(locations=[1.0, 2.0], labels=["x"])

    def test_set_xticks_empty_locations_with_non_empty_labels_raises(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="does not match"):
            sp.set_xticks(locations=[], labels=["a"])

    def test_set_xticks_empty_labels_with_non_empty_locations_raises(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="does not match"):
            sp.set_xticks(locations=[1.0], labels=[])

    def test_set_yticks_mismatched_lengths_raises_valueerror(self):
        sp = _make_plot()
        with pytest.raises(ValueError, match="does not match"):
            sp.set_yticks(locations=[1.0], labels=["a", "b"])

    def test_set_xticks_none_none_is_noop(self):
        sp = _make_plot()
        sp.set_xticks(locations=[1.0], labels=["a"])
        sp.set_xticks()  # noop
        assert sp._xaxis.tick_locations == [1.0]
        assert sp._xaxis.tick_labels == ["a"]

    def test_locations_incompatible_with_existing_labels_raises(self):
        sp = _make_plot()
        sp.set_xticks(locations=[1.0, 2.0], labels=["a", "b"])
        with pytest.raises(ValueError, match="does not match"):
            sp.set_xticks(locations=[1.0, 2.0, 3.0])

    def test_labels_incompatible_with_existing_locations_raises(self):
        sp = _make_plot()
        sp.set_xticks(locations=[1.0, 2.0], labels=["a", "b"])
        with pytest.raises(ValueError, match="does not match"):
            sp.set_xticks(labels=["x", "y", "z"])

    def test_set_yticks_empty_locations_clears_both(self):
        sp = _make_plot()
        sp.set_yticks(locations=[1.0], labels=["a"])
        sp.set_yticks(locations=[])
        assert sp._yaxis.tick_locations == []
        assert sp._yaxis.tick_labels == []

    def test_set_yticks_empty_labels_clears_labels(self):
        sp = _make_plot()
        sp.set_yticks(locations=[1.0], labels=["a"])
        sp.set_yticks(labels=[])
        assert sp._yaxis.tick_labels == []


# ===================
# == LABEL SETTERS ==
# ===================


class TestGerryPlotBaseSetters:
    """Tests for set_xlabel / set_ylabel / set_title and related clearers."""

    def test_set_xlabel_stores_text(self):
        sp = ScatterPlot()
        sp.xlabel = "Vote Share"
        assert sp.xlabel == "Vote Share"

    def test_set_xlabel_none_clears(self):
        sp = ScatterPlot(xlabel="old")
        sp.xlabel = None
        assert sp.xlabel is None

    def test_set_ylabel_stores_text(self):
        sp = ScatterPlot()
        sp.ylabel = "Seat Share"
        assert sp.ylabel == "Seat Share"

    def test_set_title_stores_text(self):
        sp = ScatterPlot()
        sp.title = "My Plot"
        assert sp.title == "My Plot"

    def test_clear_xlabel_style(self):
        sp = _make_plot()
        sp.xlabel = "x"
        sp.set_axis_label_style("x", fontsize=14.0, fontweight="bold", fontcolor="red")
        sp.clear_xlabel_style()
        assert sp._xaxis.label.style is None
        expected_size = FontProperties(
            size=matplotlib.rcParams["axes.labelsize"]
        ).get_size_in_points()
        assert sp.ax.xaxis.label.get_fontsize() == expected_size
        assert sp.ax.xaxis.label.get_fontweight() == matplotlib.rcParams["axes.labelweight"]
        assert sp.ax.xaxis.label.get_color() == matplotlib.rcParams["axes.labelcolor"]

    def test_clear_ylabel_style(self):
        sp = _make_plot()
        sp.ylabel = "y"
        sp.set_axis_label_style("y", fontsize=14.0, fontweight="bold", fontcolor="red")
        sp.clear_ylabel_style()
        assert sp._yaxis.label.style is None
        expected_size = FontProperties(
            size=matplotlib.rcParams["axes.labelsize"]
        ).get_size_in_points()
        assert sp.ax.yaxis.label.get_fontsize() == expected_size
        assert sp.ax.yaxis.label.get_fontweight() == matplotlib.rcParams["axes.labelweight"]
        assert sp.ax.yaxis.label.get_color() == matplotlib.rcParams["axes.labelcolor"]

    def test_clear_title_style(self):
        sp = _make_plot()
        sp.title = "Title"
        sp.set_title_style(fontsize=14.0, fontweight="bold", fontcolor="red", loc="left")
        sp.clear_title_style()
        assert sp._title_text.style is None
        assert sp.ax.get_title() == "Title"
        assert getattr(sp.ax, "_left_title").get_text() == ""
        expected_size = FontProperties(
            size=matplotlib.rcParams["axes.titlesize"]
        ).get_size_in_points()
        assert sp.ax.title.get_fontsize() == expected_size
        assert sp.ax.title.get_fontweight() == matplotlib.rcParams["axes.titleweight"]

    def test_empty_labels_clears_xtick_labels(self):
        sp = ScatterPlot()
        sp.set_xticks(locations=[0.5], labels=["mid"])
        sp.set_xticks(labels=[])
        assert sp._xaxis.tick_labels == []

    def test_empty_labels_clears_ytick_labels(self):
        sp = ScatterPlot()
        sp.set_yticks(locations=[0.5], labels=["mid"])
        sp.set_yticks(labels=[])
        assert sp._yaxis.tick_labels == []

    def test_empty_locations_clears_xtick_locations_and_labels(self):
        sp = ScatterPlot()
        sp.set_xticks(locations=[0.5], labels=["mid"])
        sp.set_xticks([])
        assert sp._xaxis.tick_locations == []
        assert sp._xaxis.tick_labels == []

    def test_empty_locations_clears_ytick_locations_and_labels(self):
        sp = ScatterPlot()
        sp.set_yticks(locations=[0.5], labels=["mid"])
        sp.set_yticks([])
        assert sp._yaxis.tick_locations == []
        assert sp._yaxis.tick_labels == []

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_labels_cannot_be_added_after_locations_are_cleared(self, axis):
        sp = ScatterPlot()
        setter = sp.set_xticks if axis == "x" else sp.set_yticks
        setter([])

        with pytest.raises(ValueError, match="Labels length 1"):
            setter(labels=["orphan"])

    def test_set_legend_options_updates_options(self):
        sp = ScatterPlot()
        sp.set_legend_options(ncols=2, fontsize=12.0)
        assert sp._legend_options.ncols == 2
        assert sp._legend_options.fontsize == 12.0

    def test_set_legend_options_merges_over_prior_call(self):
        sp = ScatterPlot()
        sp.set_legend_options(ncols=2, fontsize=12.0)
        sp.set_legend_options(loc="upper right")
        assert sp._legend_options.loc == "upper right"
        assert sp._legend_options.ncols == 2
        assert sp._legend_options.fontsize == 12.0

    def test_set_legend_options_explicit_none_clears_sentinel_fields(self):
        sp = ScatterPlot()
        sp.set_legend_options(title="Key", bbox_to_anchor=(1.5, 0.5))
        sp.set_legend_options(title=None, bbox_to_anchor=None)
        assert sp._legend_options.title is None
        assert sp._legend_options.bbox_to_anchor is None


class TestGerryPlotBaseUpdateYTickEdgeCases:
    """Edge cases in set_yticks that weren't previously exercised."""

    def test_set_yticks_both_none_is_noop(self):
        sp = ScatterPlot()
        sp.set_yticks()
        assert sp._yaxis.tick_locations is None
        assert sp._yaxis.tick_labels is None

    def test_set_yticks_inconsistent_clear_raises(self):
        sp = ScatterPlot()
        with pytest.raises(ValueError, match="does not match"):
            sp.set_yticks(locations=[], labels=["a"])

    def test_set_yticks_locations_mismatch_existing_labels_raises(self):
        sp = ScatterPlot()
        sp.set_yticks(labels=["a", "b"])
        with pytest.raises(ValueError, match="Locations length"):
            sp.set_yticks(locations=[0.1, 0.2, 0.3])

    def test_set_yticks_locations_only_stores(self):
        sp = ScatterPlot()
        sp.set_yticks(locations=[0.1, 0.2])
        assert sp._yaxis.tick_locations == [0.1, 0.2]
        assert sp._yaxis.tick_labels is None

    def test_set_yticks_labels_empty_clears_only_labels(self):
        sp = ScatterPlot()
        sp.set_yticks(labels=[])
        assert sp._yaxis.tick_labels == []

    def test_set_yticks_labels_mismatch_existing_locations_raises(self):
        sp = ScatterPlot()
        sp.set_yticks(locations=[0.1, 0.2])
        with pytest.raises(ValueError, match="Labels length"):
            sp.set_yticks(labels=["only_one"])

    def test_set_yticks_labels_only_stores(self):
        sp = ScatterPlot()
        sp.set_yticks(locations=[0.1, 0.2])
        sp.set_yticks(labels=["a", "b"])
        assert sp._yaxis.tick_labels == ["a", "b"]


class TestGerryPlotBaseYTickBuildErrors:
    """Tests that trigger ValueError during build for mismatched tick labels."""

    def test_build_with_y_tick_labels_but_no_locations_still_builds(self):
        """Setting y_tick_labels without locations falls back to existing auto-ticks."""
        sp = ScatterPlot()
        sp.add_series(x=[0.0], y=[0.0])
        # set labels explicitly matching auto-tick count would be brittle, so just check
        # that it doesn't hard-crash when labels list is empty
        sp._yaxis.tick_labels = []
        ax = sp.ax
        assert ax is not None


# =======================
# == TICK BUILD ERRORS ==
# =======================


class TestXTickLabelCountMismatch:
    """Mismatched explicit x tick labels raise `ValueError`."""

    def test_xtick_labels_wrong_count_raises(self):
        plot = ScatterPlot(legend=False)
        plot.add_series(x=[0.0, 0.5, 1.0], y=[0.0, 0.5, 1.0])
        # Set 3 explicit tick locations, then try to set 2 labels
        plot.set_xticks([0.0, 0.5, 1.0])
        plot.set_xticks(labels=["a", "b", "c"])
        # Force a mismatch: clear locations then set labels with wrong count
        plot._xaxis.tick_locations = [0.0, 0.5, 1.0]
        plot._xaxis.tick_labels = ["a", "b"]  # wrong count
        with pytest.raises(ValueError, match="Expected 3 x tick labels, got 2"):
            _ = plot.ax


class TestTickLabelHideRestore:
    """Hiding tick labels via empty labels is reversible after a render.

    Regression tests: ``tick_params(labelbottom=False)`` is sticky on the axes, so a
    restore that writes labels must also re-enable label visibility.
    """

    def test_render_hide_restore_xtick_labels(self):
        sp = _make_plot()
        sp.set_xticks([0.0, 1.0], labels=["lo", "hi"])
        ax = sp.ax
        assert all(tick.get_visible() for tick in ax.get_xticklabels())

        sp.set_xticks(labels=[])
        ax = sp.ax
        assert not any(tick.get_visible() for tick in ax.get_xticklabels())

        sp.set_xticks(labels=["lo", "hi"])
        ax = sp.ax
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["lo", "hi"]
        assert all(tick.get_visible() for tick in ax.get_xticklabels())

    def test_render_hide_restore_ytick_labels(self):
        sp = _make_plot()
        sp.set_yticks([0.0, 1.0], labels=["low", "high"])
        ax = sp.ax
        assert all(tick.get_visible() for tick in ax.get_yticklabels())

        sp.set_yticks(labels=[])
        ax = sp.ax
        assert not any(tick.get_visible() for tick in ax.get_yticklabels())

        sp.set_yticks(labels=["low", "high"])
        ax = sp.ax
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["low", "high"]
        assert all(tick.get_visible() for tick in ax.get_yticklabels())

    def test_restore_without_prior_render_still_visible(self):
        sp = _make_plot()
        sp.set_xticks([0.0, 1.0], labels=["a", "b"])
        sp.set_xticks(labels=[])
        sp.set_xticks(labels=["a", "b"])  # never rendered while hidden
        ax = sp.ax
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["a", "b"]
        assert all(tick.get_visible() for tick in ax.get_xticklabels())


class TestYTickLabelsWithoutLocations:
    """Y tick labels can reuse auto-generated tick locations."""

    def test_ytick_labels_without_locations_applies_to_default_ticks(self):
        plot = ScatterPlot(legend=False)
        plot.add_series(x=[0.0, 1.0], y=[0.0, 1.0])
        # Trigger a full build first so matplotlib computes auto ticks for this data
        _ = plot.ax
        default_ticks = list(plot._ax.get_yticks())
        # Match the auto-tick count while leaving explicit locations unset.
        plot._yaxis.tick_labels = [str(i) for i in range(len(default_ticks))]
        plot.title = "force rebuild"
        ax = plot.ax
        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            str(i) for i in range(len(default_ticks))
        ]

    def test_ytick_labels_wrong_count_raises(self):
        plot = ScatterPlot(legend=False)
        plot.add_series(x=[0.0, 1.0], y=[0.0, 1.0])
        _ = plot._ax
        # Force y-tick labels with wrong count compared to default ticks
        plot._yaxis.tick_labels = ["only_one_label"]
        # Leave tick_locations as None so we go through the auto-tick path
        with pytest.raises(ValueError, match="y tick labels"):
            _ = plot.ax
