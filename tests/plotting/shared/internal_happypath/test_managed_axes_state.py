"""Regression matrix for the artist-registry / managed-axes-state design.

Exercises the contract gerrytools plots maintain on a shared matplotlib
axes:

- external artist preservation across rebuilds;
- gerrytools artist non-accumulation across N rebuilds;
- most-recent-wins per managed axes unit (xlim, ylim, ticks, tick styles,
  labels, title, frame, legend);
- atomic tick / label / title units;
- store-and-claim setter semantics (labels-only ``set_xticks``);
- property-setter reclaim (``hist.title = "Foo"``);
- constructor-default vs. constructor-non-default reclaim semantics
  (including the deliberately-unsupported ``title=None`` clear via ctor);
- internal ``UNSET`` text sentinel never escaping the public API;
- legend lifecycle paths (named add_*, set_legend_options, legend
  assignment) and the "no legend handle when name is None" invariant;
- ``bind_to_ax`` reclaim-state carry, last-applied reset, and reactivation
  of yielded explicit state.

Histogram and BoxPlot are the canonical representatives.
"""

from __future__ import annotations

import inspect
from typing import get_args

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from gerrytools.plotting._axes_state import _ALL_UNITS, Unit  # noqa: E402
from gerrytools.plotting.data.boxplot import BoxPlot  # noqa: E402
from gerrytools.plotting.data.histogram import Histogram  # noqa: E402
from gerrytools.plotting.data.paintball import PaintballPlot  # noqa: E402
from gerrytools.plotting.data.scatterplot import ScatterPlot  # noqa: E402
from gerrytools.plotting.data.sealevel import SeaLevelPlot  # noqa: E402
from gerrytools.plotting.data.seatsvotes import SeatsVotesPlot  # noqa: E402
from gerrytools.plotting.data.violin import ViolinPlot  # noqa: E402
from gerrytools.plotting.utils import UNSET  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_hist() -> Histogram:
    hist = Histogram()
    hist.add_dataset([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    return hist


def _simple_box() -> BoxPlot:
    box = BoxPlot()
    box.add_dataset({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [2.0, 3.0, 4.0, 5.0, 6.0]})
    return box


def _total_artist_count(ax) -> int:
    return len(ax.patches) + len(ax.lines) + len(ax.collections) + len(ax.texts) + len(ax.images)


def _force_rebuild(plot) -> None:
    """Exercise rebuild-time ownership reconciliation without changing managed state."""
    plot._axis_needs_update = True
    plot.ax


# ---------------------------------------------------------------------------
# Category 1: external artist preservation + leak guardrails
# ---------------------------------------------------------------------------


class TestExternalArtifactPreservation:
    def test_external_text_survives_rebuild(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.text(5.0, 1.0, "external annotation")
        # Trigger a rebuild via .ax again
        hist.ax
        matching = [t for t in hist._ax.texts if t.get_text() == "external annotation"]
        assert len(matching) == 1

    def test_external_axhline_survives_rebuild(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.axhline(0.5, color="red")
        before = len(ax.lines)
        hist.ax
        # Lines may grow by one tracked overlay or not at all depending on
        # gerrytools draw paths; key is the external line is still present.
        assert any(
            l.get_color() == "red"
            for l in hist._ax.lines  # noqa: E741
        )
        assert len(hist._ax.lines) >= before

    def test_external_imshow_survives_rebuild_for_histogram(self):
        fig, ax = plt.subplots()
        ax.imshow([[1, 2], [3, 4]])
        hist = Histogram(ax=ax)
        hist.add_dataset([1, 2, 3, 4, 5])
        n_before = len(ax.images)
        hist.ax
        assert len(ax.images) == n_before

    def test_histogram_autoscale_keeps_external_collection_visible(self):
        fig, ax = plt.subplots()
        ax.scatter([100.0], [100.0])
        hist = Histogram(ax=ax)
        hist.add_dataset([1.0, 2.0, 3.0])

        x_limits = hist.ax.get_xlim()
        y_limits = hist.ax.get_ylim()

        assert x_limits[0] <= 100.0 <= x_limits[1]
        assert y_limits[0] <= 100.0 <= y_limits[1]


class TestGerrytoolsArtifactLeak:
    def test_histogram_artist_counts_stay_flat_across_rebuilds(self):
        hist = _simple_hist()
        counts = [_total_artist_count(hist.ax) for _ in range(6)]
        assert counts[0] > 0
        assert len(set(counts)) == 1, f"artist counts drift across rebuilds: {counts}"

    def test_boxplot_artist_counts_stay_flat_across_rebuilds(self):
        box = _simple_box()
        counts = [_total_artist_count(box.ax) for _ in range(6)]
        assert counts[0] > 0
        assert len(set(counts)) == 1, f"artist counts drift across rebuilds: {counts}"


class TestImplicitDefaultsUpdateWhenUntouched:
    def test_xlim_updates_when_more_data_added_and_user_has_not_touched(self):
        hist = Histogram()
        hist.add_dataset([1.0, 2.0, 3.0])
        first = hist.ax.get_xlim()
        hist.add_dataset([50.0, 60.0, 70.0])
        second = hist.ax.get_xlim()
        # New data extends well past the first dataset; default xlim should
        # widen to cover it.
        assert second[1] > first[1] + 10.0

    def test_annotation_limits_do_not_block_later_data_autoscaling(self):
        hist = Histogram()
        hist.add_dataset([1.0, 2.0, 3.0])
        hist.add_vertical_lines(100.0)
        hist.ax

        hist.add_dataset([200.0])
        left, right = hist.ax.get_xlim()

        assert left <= 200.0 <= right

    def test_annotation_outside_data_remains_visible(self):
        hist = Histogram()
        hist.add_dataset([1.0, 2.0, 3.0])
        hist.add_vertical_lines(100.0)

        left, right = hist.ax.get_xlim()

        assert left <= 100.0 <= right

    def test_out_of_range_annotation_does_not_flip_tick_ownership_external(self):
        # Regression: annotations render after the tick reconcile and can autoscale the
        # axes, so the recorded tick values went stale and the next rebuild misread the
        # shift as an external change, silently flipping x_ticks/y_ticks to "external".
        hist = Histogram()
        hist.add_dataset([1.0, 2.0, 3.0])
        hist.add_vertical_lines(100.0, name="far marker")
        hist.ax
        hist.ax
        hist.ax
        externally_owned = [
            unit for unit, state in hist._axes_state._units.items() if state.ownership == "external"
        ]
        # The user never touched the axes directly; nothing may report external.
        assert externally_owned == []


# ---------------------------------------------------------------------------
# Category 2: most-recent-wins per unit
# ---------------------------------------------------------------------------


class TestMostRecentWinsXLim:
    def test_external_xlim_beats_prior_implicit_default(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.set_xlim(-100.0, 100.0)
        hist.ax  # rebuild
        assert hist._ax.get_xlim() == (-100.0, 100.0)

    def test_explicit_gerrytools_xlim_beats_prior_external(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.set_xlim(-100.0, 100.0)
        hist.ax
        hist.set_xlim(-5.0, 5.0)
        hist.ax
        assert hist._ax.get_xlim() == (-5.0, 5.0)

    def test_external_xlim_beats_prior_explicit_after_most_recent_change(self):
        hist = _simple_hist()
        hist.set_xlim(-5.0, 5.0)
        ax = hist.ax
        ax.set_xlim(0.0, 50.0)
        _force_rebuild(hist)
        assert hist._ax.get_xlim() == (0.0, 50.0)
        assert hist._axes_state._units["x_limits"].ownership == "external"


class TestMostRecentWinsYLim:
    def test_external_ylim_survives_rebuild(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.set_ylim(0.0, 999.0)
        hist.ax
        assert hist._ax.get_ylim() == (0.0, 999.0)


class TestMostRecentWinsTitle:
    def test_external_title_beats_prior_default(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.set_title("from matplotlib")
        hist.ax
        assert hist._ax.get_title() == "from matplotlib"

    def test_explicit_title_beats_prior_external(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.set_title("from matplotlib")
        hist.ax
        hist.title = "from gerrytools"
        hist.ax
        assert hist._ax.get_title() == "from gerrytools"

    def test_external_title_at_another_location_preserves_prior_title(self):
        hist = _simple_hist()
        hist.title = "from gerrytools"
        hist.set_title_style(loc="left")
        ax = hist.ax
        ax.set_title("from matplotlib")

        _force_rebuild(hist)

        assert ax.get_title(loc="left") == "from gerrytools"
        assert ax.get_title() == "from matplotlib"

    def test_external_panel_label_preserves_center_title(self):
        hist = _simple_hist()
        hist.title = "from gerrytools"
        ax = hist.ax
        ax.set_title("(a)", loc="left")

        _force_rebuild(hist)

        assert ax.get_title(loc="left") == "(a)"
        assert ax.get_title() == "from gerrytools"


class TestMostRecentWinsFrame:
    def test_external_spine_visibility_survives_rebuild(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        hist.ax
        assert hist._ax.spines["top"].get_visible() is False
        assert hist._ax.spines["right"].get_visible() is False

    def test_explicit_frame_beats_prior_external(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.spines["top"].set_visible(False)
        hist.ax
        hist.set_frame_visibility(top=True, right=True, left=True, bottom=True)
        hist.ax
        assert hist._ax.spines["top"].get_visible() is True


# ---------------------------------------------------------------------------
# Category 3: ticks (atomic + style)
# ---------------------------------------------------------------------------


class TestTickAtomicity:
    def test_external_tick_labels_change_yields_ticks_unit(self):
        hist = _simple_hist()
        ax = hist.ax
        # Materialize a known set of locations + labels via gerrytools.
        hist.set_xticks([0.0, 5.0, 10.0], labels=["a", "b", "c"])
        hist.ax
        # User now relabels externally — should be detected as external.
        ax.set_xticks([0.0, 5.0, 10.0], labels=["X", "Y", "Z"])
        hist.ax
        labels = [t.get_text() for t in hist._ax.get_xticklabels()]
        assert labels == ["X", "Y", "Z"]


class TestTickStyleIndependent:
    def test_tick_style_obeys_most_recent_wins_independently_of_locations(self):
        hist = _simple_hist()
        # Set locations + apply a tick style
        hist.set_xticks([0.0, 5.0, 10.0])
        hist.set_tick_style("x", size=18.0, rotation=45.0)
        hist.ax
        # Externally change tick label size — this should be detected and
        # preserved on next rebuild without disturbing tick locations.
        for tlabel in hist._ax.get_xticklabels():
            tlabel.set_fontsize(7.0)
        hist.ax
        sizes = [t.get_fontsize() for t in hist._ax.get_xticklabels()]
        assert all(s == 7.0 for s in sizes)
        # Tick locations stayed the gerrytools-explicit set.
        assert list(hist._ax.get_xticks()) == [0.0, 5.0, 10.0]


class TestMinorTickStyleExternalDetection:
    def test_external_minor_tick_change_survives_rebuild_with_minor_ticktype(self):
        # Regression: the tick-style snapshot used to read major ticks only, so external
        # minor-tick changes were invisible and silently overwritten on rebuild.
        from matplotlib.colors import to_rgba

        hist = _simple_hist()
        ax = hist.ax
        ax.minorticks_on()
        hist.set_tick_style("x", tickcolor="blue", ticktype="minor")
        hist.ax
        minor_ticks = hist._ax.xaxis.get_minor_ticks()
        assert minor_ticks
        assert to_rgba(minor_ticks[0].tick1line.get_color()) == to_rgba("blue")
        # External minor-tick recolor between rebuilds must win.
        for tick in hist._ax.xaxis.get_minor_ticks():
            tick.tick1line.set_color("red")
        hist.ax
        first_minor = hist._ax.xaxis.get_minor_ticks()[0]
        assert to_rgba(first_minor.tick1line.get_color()) == to_rgba("red")

    def test_external_minor_change_does_not_yield_major_ticktype_style(self):
        # A stored major-only style must keep majors applied (unit not yielded) when only
        # minor ticks change externally.
        from matplotlib.colors import to_rgba

        hist = _simple_hist()
        ax = hist.ax
        ax.minorticks_on()
        hist.set_tick_style("x", tickcolor="blue", ticktype="major")
        hist.ax
        for tick in hist._ax.xaxis.get_minor_ticks():
            tick.tick1line.set_color("red")
        hist.ax
        assert hist._axes_state.is_reclaimed("x_tick_style")
        major_colors = {
            to_rgba(tick.tick1line.get_color()) for tick in hist._ax.xaxis.get_major_ticks()
        }
        assert major_colors == {to_rgba("blue")}


class TestExplicitTicksSurviveLegendRedraw:
    def test_explicit_xticks_survive_legend_apply(self):
        hist = _simple_hist()
        hist.set_xticks([0.0, 5.0, 10.0], labels=["a", "b", "c"])
        # Force a legend render.
        hist.legend = True
        hist.ax
        assert list(hist._ax.get_xticks()) == [0.0, 5.0, 10.0]
        assert [t.get_text() for t in hist._ax.get_xticklabels()] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Category 4: legend lifecycle
# ---------------------------------------------------------------------------


class TestLegendLifecycle:
    def test_legend_placed_before_the_first_build_survives_it(self):
        from matplotlib.lines import Line2D

        _, ax = plt.subplots()
        hist = Histogram(ax=ax)
        hist.add_dataset([1, 2, 3])  # unnamed, so gerrytools never claims the legend
        ax.legend(handles=[Line2D([0], [0], label="user")])
        user_legend = ax.get_legend()

        hist.ax  # first build

        assert ax.get_legend() is user_legend

    def test_external_legend_survives_rebuild(self):
        hist = _simple_hist()
        hist.legend = False  # disable gerrytools legend
        ax = hist.ax
        ax.plot([0, 1], [0, 1], label="external")
        ax.legend()
        external_legend = ax.get_legend()
        assert external_legend is not None
        hist.ax  # rebuild
        # Same legend object should still be on the axes (we didn't touch it).
        assert hist._ax.get_legend() is external_legend

    def test_external_replacement_legend_survives_rebuild(self):
        # Regression: the stale gerrytools legend must not be removed once the user
        # replaces it (matplotlib's legend-remove hook nulls ``ax.legend_``
        # unconditionally, which would destroy the external legend).
        hist = _simple_hist()
        hist.legend = True
        hist.add_dataset([1, 2, 3], name="A")
        ax = hist.ax
        assert ax.get_legend() is not None
        ax.plot([0, 1], [0, 1], label="external")
        external_legend = ax.legend()
        hist.ax  # rebuild
        assert hist._ax.get_legend() is external_legend

    def test_set_legend_options_reclaims_legend_unit(self):
        hist = _simple_hist()
        hist.legend = True
        hist.add_dataset([1, 2, 3], name="A")
        hist.ax  # first render places a gerrytools legend
        first_legend = hist._ax.get_legend()
        hist.set_legend_options(loc="upper right")
        hist.ax
        new_legend = hist._ax.get_legend()
        assert new_legend is not None
        assert new_legend is not first_legend  # legend was replaced

    def test_include_legend_false_removes_gerrytools_legend(self):
        hist = _simple_hist()
        hist.legend = True
        hist.add_dataset([1, 2, 3], name="A")
        hist.ax
        assert hist._ax.get_legend() is not None
        hist.legend = False
        hist.ax
        assert hist._ax.get_legend() is None

    def test_rebinding_back_remembers_which_legend_gerrytools_placed(self):
        _, first_ax = plt.subplots()
        _, second_ax = plt.subplots()
        hist = Histogram(ax=first_ax, legend=True)
        hist.add_dataset([1, 2, 3], name="A")
        assert hist.ax.get_legend() is not None

        hist.bind_to_ax(second_ax)
        assert second_ax.get_legend() is not None
        hist.legend = False
        hist.bind_to_ax(first_ax)

        assert first_ax.get_legend() is None

    def test_include_legend_assignment_reclaims_legend_unit(self):
        hist = _simple_hist()
        hist.add_dataset([1, 2, 3], name="A")
        hist.ax
        # Initially gerrytools owns the legend after placing it.
        hist.legend = False
        # Re-enable; gerrytools should put it back.
        hist.legend = True
        hist.ax
        assert hist._ax.get_legend() is not None

    def test_no_legend_when_include_legend_false_and_no_external_legend(self):
        # With legend=False, gerrytools never places a legend even
        # when handles exist.
        hist = Histogram(legend=False)
        hist.add_dataset([1, 2, 3], name="A")
        hist.ax
        assert hist._ax.get_legend() is None

    def test_user_legend_survives_set_legend_options_with_zero_handles(self):
        # Regression: with zero handles and a store-and-claimed legend unit, the
        # rebuild used to remove a legend gerrytools never placed.
        plot = ScatterPlot()
        plot.add_series(x=[0.1, 0.2], y=[0.3, 0.4])  # unnamed → zero legend handles
        ax = plot.ax
        ax.plot([0, 1], [0, 1], label="external")
        user_legend = ax.legend()
        plot.set_legend_options(loc="upper right")
        plot.ax
        assert plot._ax.get_legend() is user_legend


# ---------------------------------------------------------------------------
# Category 5: bind_to_ax + scenario E + sentinel
# ---------------------------------------------------------------------------


class TestBindToAx:
    def test_explicit_state_survives_bind_to_new_axes(self):
        hist = _simple_hist()
        hist.set_xlim(-5.0, 5.0)
        hist.ax
        fig2, ax2 = plt.subplots()
        hist.bind_to_ax(ax2)
        hist.ax
        assert hist._ax is ax2
        assert hist._ax.get_xlim() == (-5.0, 5.0)

    def test_explicit_state_wins_over_preset_new_axes(self):
        hist = _simple_hist()
        hist.set_xlim(-5.0, 5.0)
        hist.ax
        # New axes already has a different xlim.
        fig2, ax2 = plt.subplots()
        ax2.set_xlim(0.0, 50.0)
        hist.bind_to_ax(ax2)
        hist.ax
        # gerrytools reclaim wins over pre-existing axes state.
        assert hist._ax.get_xlim() == (-5.0, 5.0)

    def test_yielded_explicit_reactivates_on_rebind(self):
        hist = _simple_hist()
        hist.set_xlim(-5.0, 5.0)
        ax = hist.ax
        # User externally overrides — gerrytools yields x_limits.
        ax.set_xlim(0.0, 20.0)
        hist.ax
        assert hist._ax.get_xlim() == (0.0, 20.0)
        # Now rebind: explicit gerrytools configuration reactivates.
        fig2, ax2 = plt.subplots()
        hist.bind_to_ax(ax2)
        hist.ax
        assert hist._ax.get_xlim() == (-5.0, 5.0)

    def test_old_axes_output_survives_rebind(self):
        hist = _simple_hist()
        old_ax = hist.ax
        n_before = _total_artist_count(old_ax)
        assert n_before > 0
        fig2, ax2 = plt.subplots()
        hist.bind_to_ax(ax2)
        hist.ax
        # Old axes content was not removed.
        assert _total_artist_count(old_ax) >= n_before

    def test_round_trip_does_not_misclassify_own_limits_and_ticks_as_external(self):
        box = _simple_box()
        original_ax = box.ax
        _, other_ax = plt.subplots()
        box.bind_to_ax(other_ax)
        box.bind_to_ax(original_ax)

        box.add_dataset({"C": [100.0, 110.0, 120.0]}, add_extra_labels=True)
        rebound = box.ax

        assert rebound.get_ylim()[1] >= 120.0
        assert "C" in [label.get_text() for label in rebound.get_xticklabels()]
        assert box._axes_state._units["x_ticks"].ownership != "external"
        assert box._axes_state._units["y_limits"].ownership != "external"


class TestScenarioE:
    """Pre-configured axes state is preserved at bind time."""

    def test_preset_xlim_preserved_at_bind(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0.0, 50.0)
        hist = Histogram(ax=ax)
        assert hist._axes_state._units["x_limits"].ownership == "external"
        hist.add_dataset([1, 2, 3])
        hist.ax
        assert ax.get_xlim() == (0.0, 50.0)

    def test_preset_title_preserved_when_constructor_title_is_none(self):
        fig, ax = plt.subplots()
        ax.set_title("user title")
        hist = Histogram(ax=ax)  # title=None is the Python default → no opinion
        hist.add_dataset([1, 2, 3])
        hist.ax
        assert ax.get_title() == "user title"


class TestUnsetTextSentinelHidden:
    def test_sentinel_never_appears_in_public_signature(self):
        sig = inspect.signature(Histogram.__init__)
        for param in sig.parameters.values():
            assert param.default is not UNSET

    def test_sentinel_never_appears_in_property_getter_output(self):
        # The sentinel is stored on _title but the public getter must map it
        # back to None.
        hist = Histogram()  # no title
        assert hist.title is None  # not the sentinel
        assert hist.xlabel is None
        assert hist.ylabel is None


# ---------------------------------------------------------------------------
# Category 6: store-and-claim + property setters
# ---------------------------------------------------------------------------


class TestStoreAndClaim:
    def test_set_xticks_labels_claims_unit_at_call_time(self):
        hist = _simple_hist()
        hist.ax  # first render — tick locations come from matplotlib default
        # Store-and-claim path: labels-only, no locations.
        existing_locations = hist._ax.get_xticks().tolist()
        labels = [f"L{i}" for i in range(len(existing_locations))]
        hist.set_xticks(labels=labels)
        hist.ax
        rendered = [t.get_text() for t in hist._ax.get_xticklabels()]
        assert rendered == labels

    def test_store_and_claim_survives_intervening_external_xtick_change(self):
        hist = _simple_hist()
        hist.set_xticks([0.0, 5.0, 10.0])
        hist.ax
        # The labels-only set_xticks claims x_ticks ownership at this point.
        hist.set_xticks(labels=["P", "Q", "R"])
        # User mutates ticks externally between the claim and the next rebuild.
        # The reclaim happened at the store call → gerrytools labels win.
        hist._ax.set_xticklabels(["x", "y", "z"])
        hist.ax
        assert [t.get_text() for t in hist._ax.get_xticklabels()] == ["P", "Q", "R"]


class TestPropertySetterReclaim:
    def test_title_property_setter_reclaims_title_unit(self):
        hist = _simple_hist()
        hist.title = "Hello"
        hist.ax
        assert hist._ax.get_title() == "Hello"
        # External change wins on next rebuild.
        hist._ax.set_title("External")
        hist.ax
        assert hist._ax.get_title() == "External"
        # gerrytools setter wins again.
        hist.title = "Again"
        hist.ax
        assert hist._ax.get_title() == "Again"

    def test_constructor_title_reclaims_at_construction(self):
        hist = Histogram(title="My Plot")
        hist.add_dataset([1, 2, 3])
        ax = hist.ax
        assert ax.get_title() == "My Plot"

    def test_constructor_title_none_preserves_preset_title(self):
        """``title=None`` means "no opinion" and preserves a preset title.

        Clearing happens after construction through the property setter.
        """
        fig, ax = plt.subplots()
        ax.set_title("preset")
        hist = Histogram(ax=ax, title=None)
        hist.add_dataset([1, 2, 3])
        hist.ax
        assert ax.get_title() == "preset"

    def test_post_construction_title_none_clears_title(self):
        hist = Histogram(title="Hello")
        hist.add_dataset([1, 2, 3])
        hist.ax
        assert hist._ax.get_title() == "Hello"
        hist.title = None
        hist.ax
        # ``set_title(None)`` is matplotlib's clear path → empty string.
        assert hist._ax.get_title() == ""


class TestStyleOnlySetterReclaimsAtomicUnit:
    def test_title_style_setter_reclaims_title_unit(self):
        hist = _simple_hist()
        hist.title = "Hello"
        hist.ax
        # Apply a style — atomic with text under the same managed unit.
        hist.set_title_style(fontsize=22.0, fontweight="bold")
        hist.ax
        title_artist = hist._ax.title
        assert title_artist.get_text() == "Hello"
        assert title_artist.get_fontsize() == 22.0

    def test_xaxis_label_style_setter_reclaims_xlabel_unit(self):
        hist = _simple_hist()
        hist.xlabel = "X"
        hist.ax
        hist.set_axis_label_style("x", fontsize=18.0)
        hist.ax
        assert hist._ax.xaxis.label.get_text() == "X"
        assert hist._ax.xaxis.label.get_fontsize() == 18.0


# ---------------------------------------------------------------------------
# Category 7: managed-unit vocabulary stays in sync with the snapshot
# ---------------------------------------------------------------------------


class TestUnitVocabularySync:
    def test_unit_literal_matches_snapshot_derived_all_units(self):
        # ``_ALL_UNITS`` is derived from ``_AxesSnapshot`` fields; the ``Unit``
        # Literal must name exactly the same vocabulary.
        assert set(get_args(Unit)) == set(_ALL_UNITS)


# ---------------------------------------------------------------------------
# Category 8: additional managed-state edge cases
# ---------------------------------------------------------------------------


class TestScaleMostRecentWins:
    """xscale and yscale obey most-recent-wins.

    Gerrytools data plots do not set a matplotlib axis scale, so the only
    user-facing path is direct ``ax.set_xscale(...)`` / ``ax.set_yscale(...)``.
    The contract: gerrytools records the scale as a default each rebuild,
    detects an external mismatch, and preserves the external scale on
    subsequent rebuilds.
    """

    def test_external_xscale_change_survives_rebuild(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.set_xscale("log")
        hist.ax
        assert hist._ax.get_xscale() == "log"

    def test_external_yscale_change_survives_rebuild(self):
        hist = _simple_hist()
        ax = hist.ax
        ax.set_yscale("log")
        hist.ax
        assert hist._ax.get_yscale() == "log"

    def test_xscale_external_yielded_after_first_external_change(self):
        # After the user flips the scale, subsequent gerrytools rebuilds
        # must NOT silently revert to matplotlib's default ("linear").
        hist = _simple_hist()
        ax = hist.ax
        ax.set_xscale("log")
        for _ in range(3):
            hist.ax
        assert hist._ax.get_xscale() == "log"


class TestNamedAddReclaimsLegend:
    """A named ``add_*`` reclaims the legend; ``add_*(name=None)`` does not.

    A user-supplied ``name`` signals that the resulting handle belongs in the
    legend, so gerrytools owns the legend slot from that point on. ``name=None``
    (the Python default) means "no opinion" and leaves legend ownership alone.
    """

    def test_named_add_histogram_reclaims_legend_unit(self):
        hist = _simple_hist()
        hist.ax  # initial render (no explicit named adds yet)
        # Build a fresh plot and add a named histogram.
        hist2 = Histogram()
        assert not hist2._axes_state.is_reclaimed("legend")
        hist2.add_dataset([1, 2, 3], name="A")
        assert hist2._axes_state.is_reclaimed("legend")

    def test_unnamed_add_histogram_does_not_reclaim_legend(self):
        hist = Histogram()
        assert not hist._axes_state.is_reclaimed("legend")
        hist.add_dataset([1, 2, 3])  # name=None default
        assert not hist._axes_state.is_reclaimed("legend")

    def test_named_add_boxplot_reclaims_legend_unit(self):
        box = BoxPlot()
        assert not box._axes_state.is_reclaimed("legend")
        box.add_dataset({"A": [1, 2, 3]}, name="series-A")
        assert box._axes_state.is_reclaimed("legend")

    def test_unnamed_add_boxplot_does_not_reclaim_legend(self):
        box = BoxPlot()
        assert not box._axes_state.is_reclaimed("legend")
        box.add_dataset({"A": [1, 2, 3]})
        assert not box._axes_state.is_reclaimed("legend")


_NAMED_ADD_CASES = [
    pytest.param(
        Histogram,
        lambda plot: plot.add_points_above([1.0, 2.0], name="points"),
        id="Histogram.add_points_above",
    ),
    pytest.param(
        BoxPlot,
        lambda plot: plot.add_pointset({"A": 1.0}, name="points"),
        id="BoxPlot.add_pointset",
    ),
    pytest.param(
        ViolinPlot,
        lambda plot: plot.add_dataset({"A": [1.0, 2.0]}, name="violins"),
        id="ViolinPlot.add_dataset",
    ),
    pytest.param(
        SeaLevelPlot,
        lambda plot: plot.add_dataset({"A": 0.5, "B": 0.7}, name="levels"),
        id="SeaLevelPlot.add_dataset",
    ),
    pytest.param(
        SeatsVotesPlot,
        lambda plot: plot.add_election([0.4, 0.6], name="curve"),
        id="SeatsVotesPlot.add_election",
    ),
    pytest.param(
        SeatsVotesPlot,
        lambda plot: plot.add_proportionality_line(name="proportionality"),
        id="SeatsVotesPlot.add_proportionality_line",
    ),
    pytest.param(
        SeatsVotesPlot,
        lambda plot: plot.add_efficiency_gap_line(name="efficiency-gap"),
        id="SeatsVotesPlot.add_efficiency_gap_line",
    ),
    pytest.param(
        SeatsVotesPlot,
        lambda plot: plot.add_custom_line(
            2.0, linecolor="black", linestyle="-", linewidth=1.0, name="custom"
        ),
        id="SeatsVotesPlot.add_custom_line",
    ),
    pytest.param(
        PaintballPlot,
        lambda plot: plot.add_lines_with_slope([1.0], name="guide"),
        id="PaintballPlot.add_lines_with_slope",
    ),
    pytest.param(
        ScatterPlot,
        lambda plot: plot.add_series(x=[0.1], y=[0.2], name="points"),
        id="ScatterPlot.add_series",
    ),
    pytest.param(
        ScatterPlot,
        lambda plot: plot.add_point(0.1, 0.2, "point"),
        id="ScatterPlot.add_point",
    ),
]

# Same calls without a name/label. ``ScatterPlot.add_point`` is excluded (its label parameter is
# required) and the SeatsVotesPlot/PaintballPlot guide-line wrappers with non-None default names are
# represented by their underlying named-line methods.
_UNNAMED_ADD_CASES = [
    pytest.param(
        Histogram,
        lambda plot: plot.add_points_above([1.0, 2.0]),
        id="Histogram.add_points_above",
    ),
    pytest.param(
        BoxPlot,
        lambda plot: plot.add_pointset({"A": 1.0}),
        id="BoxPlot.add_pointset",
    ),
    pytest.param(
        ViolinPlot,
        lambda plot: plot.add_dataset({"A": [1.0, 2.0]}),
        id="ViolinPlot.add_dataset",
    ),
    pytest.param(
        SeaLevelPlot,
        lambda plot: plot.add_dataset({"A": 0.5, "B": 0.7}),
        id="SeaLevelPlot.add_dataset",
    ),
    pytest.param(
        SeatsVotesPlot,
        lambda plot: plot.add_election([0.4, 0.6]),
        id="SeatsVotesPlot.add_election",
    ),
    pytest.param(
        SeatsVotesPlot,
        lambda plot: plot.add_proportionality_line(),
        id="SeatsVotesPlot.add_proportionality_line",
    ),
    pytest.param(
        SeatsVotesPlot,
        lambda plot: plot.add_efficiency_gap_line(),
        id="SeatsVotesPlot.add_efficiency_gap_line",
    ),
    pytest.param(
        SeatsVotesPlot,
        lambda plot: plot.add_custom_line(2.0, linecolor="black", linestyle="-", linewidth=1.0),
        id="SeatsVotesPlot.add_custom_line",
    ),
    pytest.param(
        PaintballPlot,
        lambda plot: plot.add_lines_with_slope([1.0]),
        id="PaintballPlot.add_lines_with_slope",
    ),
    pytest.param(
        ScatterPlot,
        lambda plot: plot.add_series(x=[0.1], y=[0.2]),
        id="ScatterPlot.add_series",
    ),
]


class TestNamedAddReclaimsLegendAcrossPlotTypes:
    """Named and unnamed ``add_*`` calls follow the legend-ownership contract.

    Same contract as ``TestNamedAddReclaimsLegend``: a user-supplied name
    (or label, for ScatterPlot) reclaims the legend unit; omitting it leaves
    legend ownership alone.
    """

    @pytest.mark.parametrize(("plot_factory", "named_add"), _NAMED_ADD_CASES)
    def test_named_add_reclaims_legend_unit(self, plot_factory, named_add):
        plot = plot_factory()
        assert not plot._axes_state.is_reclaimed("legend")
        named_add(plot)
        assert plot._axes_state.is_reclaimed("legend")

    @pytest.mark.parametrize(("plot_factory", "unnamed_add"), _UNNAMED_ADD_CASES)
    def test_unnamed_add_does_not_reclaim_legend(self, plot_factory, unnamed_add):
        plot = plot_factory()
        unnamed_add(plot)
        assert not plot._axes_state.is_reclaimed("legend")


class TestExternalLegendReplacedByNamedAdd:
    """A later named ``add_*`` call replaces an external legend.

    Order: external legend placed → named ``add_*`` reclaims legend →
    next rebuild's ``_apply_legend`` removes the external legend and places
    a gerrytools-owned one.
    """

    def test_named_add_replaces_external_legend(self):
        hist = _simple_hist()
        hist.legend = True
        ax = hist.ax
        # User places an external legend.
        ax.plot([0, 1], [0, 1], label="external")
        ax.legend()
        external_legend = ax.get_legend()
        assert external_legend is not None
        # User calls a named add — should reclaim legend and replace on rebuild.
        hist.add_dataset([10, 20, 30], name="gerrytools-series")
        hist.ax
        new_legend = hist._ax.get_legend()
        assert new_legend is not None
        assert new_legend is not external_legend


class TestImplicitTicksSelfCorrect:
    """Implicit ticks self-correct after restored external limits.

    Gerrytools data plots do not set explicit tick locations by default; they
    let matplotlib's locator pick. After an external xlim that drifts past
    the original data range, the next rebuild restores those limits and the
    locator should recompute tick positions consistent with the new limits.
    """

    def test_implicit_xticks_track_restored_external_xlim(self):
        hist = _simple_hist()
        ax = hist.ax
        # Push xlim well past the data range.
        ax.set_xlim(-50.0, 50.0)
        hist.ax
        # xlim should be preserved (most-recent-wins, external).
        assert hist._ax.get_xlim() == (-50.0, 50.0)
        # Implicit-default ticks should cover the new range — i.e. at least
        # one tick should sit in the extended negative region and at least
        # one in the extended positive region.
        ticks = list(hist._ax.get_xticks())
        assert any(t < -5.0 for t in ticks), f"no tick in extended negative range: {ticks}"
        assert any(t > 15.0 for t in ticks), f"no tick in extended positive range: {ticks}"


class TestUnobservableNoOpLimitation:
    """Pin the unobservable-no-op limitation.

    If a user explicitly sets a managed unit to a value identical to its
    matplotlib default, gerrytools cannot distinguish that from a fresh
    untouched axes. The bind-time signal compares to the matplotlib default,
    so setting the default at bind time looks untouched. This is a fundamental
    constraint of matplotlib's state model; the test pins the behavior so it is
    not mistaken for a bug.
    """

    def test_xlim_explicitly_set_to_default_is_unobservable_at_bind(self):
        # Default xlim is (0, 1) on a fresh axes; setting to the same value
        # leaves autoscalex_on=False but the value-comparison signal alone
        # cannot tell the difference from "fresh axes" — and we use the
        # autoscale flag, which DOES catch this. So this specific limit case
        # is actually observable. The unobservable case is the matplotlib
        # default itself.
        fig, ax = plt.subplots()
        # Setting xlabel to "" (the matplotlib default) is the canonical
        # unobservable case.
        ax.set_xlabel("")
        hist = Histogram(ax=ax)
        hist.add_dataset([1, 2, 3])
        # On rebuild, gerrytools treats this as "no opinion" and may set
        # its own xlabel default later. The user's explicit-empty intent is
        # not distinguishable from constructor-omitted.
        hist.ax
        # The xlabel remains empty either way; we just document that the
        # ownership of this unit is not "external" — it stayed "unclaimed"
        # because the bind-time comparison matched the default.
        assert hist._axes_state.is_reclaimed("x_label") is False
        # And importantly, the unit was NOT classified as external:
        # the only way to verify is to check that gerrytools is willing to
        # apply its own xlabel here without yielding.
        hist.xlabel = "from gerrytools"
        hist.ax
        assert hist._ax.xaxis.label.get_text() == "from gerrytools"


class TestConstructorDefaultDoesNotReclaim:
    """``Histogram()`` without constructor arguments does not reclaim the title.

    The constructor rule: a Python default constructor arg means "no opinion"
    and must not promote into explicit gerrytools ownership. The complement
    of ``test_constructor_title_reclaims_at_construction`` above.
    """

    def test_default_construction_does_not_reclaim_title(self):
        hist = Histogram()
        assert not hist._axes_state.is_reclaimed("title")

    def test_default_construction_does_not_reclaim_xlabel(self):
        hist = Histogram()
        assert not hist._axes_state.is_reclaimed("x_label")

    def test_default_construction_does_not_reclaim_ylabel(self):
        hist = Histogram()
        assert not hist._axes_state.is_reclaimed("y_label")

    def test_default_construction_does_not_reclaim_legend(self):
        hist = Histogram()
        assert not hist._axes_state.is_reclaimed("legend")

    def test_explicit_constructor_title_does_reclaim(self):
        hist = Histogram(title="explicit")
        assert hist._axes_state.is_reclaimed("title")
