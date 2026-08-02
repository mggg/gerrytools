import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data.seatsvotes import SeatsVotesPlot
from tests.plotting._typing_utils import as_any


def _simple_sv():
    """Create a SeatsVotesPlot with one dataset added."""
    sv = SeatsVotesPlot()
    sv.add_election([0.3, 0.6, 0.55], name="SEN20")
    return sv


# ===========================
# == CONSTRUCTION DEFAULTS ==
# ===========================
class TestAddSeatVotesData:
    def test_add_with_vote_shares_only(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.3, 0.6, 0.55])
        assert len(sv._sv_data_list) == 1

    def test_add_with_vote_counts_and_total_counts(self):
        sv = SeatsVotesPlot()
        sv.add_election([300, 600, 550], total_votes=[1000, 1000, 1000])
        assert len(sv._sv_data_list) == 1

    def test_vote_shares_out_of_range_without_total_raises_valueerror(self):
        sv = SeatsVotesPlot()
        with pytest.raises(ValueError, match="vote shares"):
            sv.add_election([0.3, 1.5, 0.55])

    def test_negative_vote_shares_without_total_raises_valueerror(self):
        sv = SeatsVotesPlot()
        with pytest.raises(ValueError, match="vote shares"):
            sv.add_election([-0.1, 0.6, 0.55])

    def test_negative_vote_counts_with_total_raise_at_add_time(self):
        sv = SeatsVotesPlot()

        with pytest.raises(ValueError, match="target_party_vote_shares cannot contain negative"):
            sv.add_election([-5.0], total_votes=[100.0])

    def test_auto_name_default(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.5, 0.6])
        assert sv._sv_data_list[0].name == "Election Seats-Votes Curve"

    def test_explicit_name(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.5, 0.6], name="GOV22")
        assert sv._sv_data_list[0].name == "GOV22"

    def test_auto_markerlabel_default(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.5, 0.6])
        assert sv._sv_data_list[0].marker_label == "Election Result"

    def test_explicit_markerlabel(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.5, 0.6], marker_label="2020 Outcome")
        assert sv._sv_data_list[0].marker_label == "2020 Outcome"

    def test_linecolor_defaults_to_standard_color(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.5, 0.6])
        assert sv._sv_data_list[0].line_style.linecolor == sv.standard_election_color

    def test_markerfacecolor_defaults_to_standard_color(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.5, 0.6])
        assert sv._sv_data_list[0].marker_style.markerfacecolor == sv.standard_marker_color

    def test_omitted_markeredgecolor_keeps_default_zero_width(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.5, 0.6])

        assert sv._sv_data_list[0].marker_style.markeredgewidth == 0.0

    def test_selected_markeredgecolor_gets_default_visible_width(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.5, 0.6], markeredgecolor="black")

        assert sv._sv_data_list[0].marker_style.markeredgewidth == 0.8

    def test_custom_linecolor(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.5, 0.6], linecolor="red")
        # Kwargs now merge through SeatsVotesLineOptions, which normalizes colors to hex,
        # matching the line_options code path.
        assert sv._sv_data_list[0].line_style.linecolor == "#ff0000"

    def test_explicit_none_colors_remove_curve_and_marker(self):
        sv = SeatsVotesPlot()
        sv.add_election(
            [0.5, 0.6],
            linecolor=None,
            markerfacecolor=None,
            markeredgecolor=None,
        )

        election = sv._sv_data_list[0]
        assert election.line_style.linecolor == "none"
        assert election.marker_style.markerfacecolor == "none"
        assert election.marker_style.markeredgecolor == "none"
        assert election.marker_style.markeredgewidth == 0.0

    def test_none_standard_colors_remove_default_curve_and_marker(self):
        sv = SeatsVotesPlot()
        sv.standard_election_color = None
        sv.standard_marker_color = None
        sv.add_election([0.5, 0.6])

        election = sv._sv_data_list[0]
        assert election.line_style.linecolor == "none"
        assert election.marker_style.markerfacecolor == "none"
        _ = sv.ax

    def test_add_multiple_datasets(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.3, 0.6], name="A")
        sv.add_election([0.4, 0.7], name="B")
        assert len(sv._sv_data_list) == 2


# ==================
# == CUSTOM LINES ==
# ==================


class TestSeatsVotesLines:
    def test_add_proportionality_line(self):
        sv = SeatsVotesPlot()
        sv.add_proportionality_line()
        assert len(sv._named_lines) == 1
        assert sv._named_lines["Proportionality"].lines[0].slope == 1.0
        assert sv._named_lines["Proportionality"].lines[0].label == "Proportionality"

    def test_add_efficiency_gap_line(self):
        sv = SeatsVotesPlot()
        sv.add_efficiency_gap_line()
        assert len(sv._named_lines) == 1
        assert sv._named_lines["Efficiency Gap"].lines[0].slope == 2.0
        assert sv._named_lines["Efficiency Gap"].lines[0].label == "Efficiency Gap"

    def test_add_custom_line(self):
        sv = SeatsVotesPlot()
        sv.add_custom_line(3.0, linecolor="red", linestyle="--", linewidth=1.5)
        assert len(sv._lines) == 1
        assert sv._lines[0].slope == 3.0

    def test_custom_line_label_kwarg_is_gone(self):
        sv = SeatsVotesPlot()
        with pytest.raises(TypeError, match="label"):
            sv.add_custom_line(
                1.0,
                linecolor="red",
                linestyle="-",
                linewidth=1.0,
                **as_any({"label": "bar"}),
            )

    def test_custom_line_name_only_sets_label(self):
        sv = SeatsVotesPlot()
        sv.add_custom_line(1.0, linecolor="red", linestyle="-", linewidth=1.0, name="myline")
        assert sv._named_lines["myline"].lines[0].label == "myline"

    def test_proportionality_line_custom_name(self):
        sv = SeatsVotesPlot()
        sv.add_proportionality_line(name="1:1")
        assert sv._named_lines["1:1"].lines[0].label == "1:1"


# ========================
# == CROSSHAIR SETTINGS ==
# ========================


class TestSeatsVotesLegend:
    def test_legend_includes_sv_curve_handle(self):
        sv = _simple_sv()
        handles = sv._legend_handles
        labels = [h.get_label() for h in handles]
        assert "SEN20" in labels

    def test_legend_includes_marker_handle_when_shown(self):
        sv = _simple_sv()
        handles = sv._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Election Result" in labels

    def test_legend_excludes_marker_handle_when_hidden(self):
        sv = _simple_sv()
        sv.display_election_markers(False)
        handles = sv._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Election Result" not in labels

    def test_legend_includes_line_handles_when_shown(self):
        sv = _simple_sv()
        sv.add_proportionality_line()
        handles = sv._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Proportionality" in labels

    def test_legend_excludes_line_handles_when_hidden(self):
        sv = _simple_sv()
        sv.add_proportionality_line()
        sv.display_additional_lines_in_legend(False)
        handles = sv._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Proportionality" not in labels

    def test_unlabeled_lines_excluded_from_legend(self):
        sv = _simple_sv()
        sv.add_custom_line(1.5, linecolor="red", linestyle="-", linewidth=1.0)  # no label
        line_handles = sv._slope_line_legend_handles()
        assert len(line_handles) == 0


# ==================================
# == NAMED ANNOTATIONS IN LEGENDS ==
# ==================================
class TestNamedAnnotationLegendHandles:
    def test_named_vline_and_band_appear_in_legend_handles(self):
        sv = _simple_sv()
        sv.add_proportionality_line()
        sv.add_vertical_lines(0.5, name="Majority")
        sv.add_vertical_band(0.45, 0.55, name="Competitive")
        labels = [handle.get_label() for handle in sv._legend_handles]
        # Datasets first, slope lines next, named annotations last.
        assert labels[0] == "SEN20"
        assert (
            labels.index("Proportionality") < labels.index("Majority") < labels.index("Competitive")
        )

    def test_legend_unchanged_without_named_annotations(self):
        sv = _simple_sv()
        sv.add_proportionality_line()
        expected = [handle.get_label() for handle in sv._dataset_legend_handles()] + [
            "Proportionality"
        ]
        labels = [handle.get_label() for handle in sv._legend_handles]
        assert labels == expected

    def test_paintball_named_vline_in_legend_handles(self):
        from gerrytools.plotting.data.paintball import PaintballPlot

        plot = PaintballPlot()
        plot.add_seats_votes_data([0.4, 0.5, 0.6], [0.3, 0.5, 0.7])
        plot.add_efficiency_gap_line()
        plot.add_vertical_lines(0.5, name="Majority")
        labels = [handle.get_label() for handle in plot._legend_handles]
        assert "Majority" in labels
        assert labels.index("Efficiency Gap") < labels.index("Majority")


# ==========================
# == ADD-TIME VALIDATION  ==
# ==========================


class TestAddElectionFiniteness:
    """Non-finite vote inputs fail at add time, matching the fail-at-add convention."""

    def test_nan_vote_share_raises_at_add_time(self):
        sv = SeatsVotesPlot()
        with pytest.raises(ValueError, match="finite"):
            sv.add_election([0.4, float("nan"), 0.6])

    def test_infinite_vote_share_raises_at_add_time(self):
        sv = SeatsVotesPlot()
        with pytest.raises(ValueError, match="finite"):
            sv.add_election([0.4, float("inf"), 0.6])

    def test_nan_total_votes_raises_at_add_time(self):
        sv = SeatsVotesPlot()
        with pytest.raises(ValueError, match="finite"):
            sv.add_election([300, 600], total_votes=[1000, float("nan")])


# ========================
# == LEGEND DEDUP RULES ==
# ========================


class TestSeatsVotesLegendDedup:
    """Identical (style, name) pairs collapse to one legend handle; distinct pairs stay."""

    def test_identical_curve_style_and_name_collapse(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.3, 0.6], name="E")
        sv.add_election([0.4, 0.7], name="E")
        assert len(sv._get_sv_curve_legend_handles()) == 1

    def test_identical_marker_style_and_label_collapse(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.3, 0.6], marker_label="Result")
        sv.add_election([0.4, 0.7], marker_label="Result")
        assert len(sv._get_sv_marker_legend_handles()) == 1

    def test_distinct_style_or_name_keeps_separate_handles(self):
        sv = SeatsVotesPlot()
        sv.add_election([0.3, 0.6], name="E")
        sv.add_election([0.4, 0.7], name="E", linecolor="red")  # same name, new style
        sv.add_election([0.2, 0.5], name="F")  # new name, default style
        assert len(sv._get_sv_curve_legend_handles()) == 3
