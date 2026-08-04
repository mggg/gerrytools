import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data.seatsvotes import SeatsVotesPlot


def _simple_sv():
    """Create a SeatsVotesPlot with one dataset added."""
    sv = SeatsVotesPlot()
    sv.add_election([0.3, 0.6, 0.55], name="SEN20")
    return sv


# ===========================
# == CONSTRUCTION DEFAULTS ==
# ===========================
class TestSeatsVotesConstruction:
    def test_default_figure_size_is_square(self):
        sv = SeatsVotesPlot()
        assert sv.fig.get_size_inches()[0] == sv.fig.get_size_inches()[1]

    def test_default_axis_limits_are_unit_square(self):
        sv = SeatsVotesPlot()
        assert sv._xaxis.limits == (0.0, 1.0)
        assert sv._yaxis.limits == (0.0, 1.0)

    def test_crosshair_enabled_by_default(self):
        sv = SeatsVotesPlot()
        assert sv._crosshair_style is not None

    def test_default_election_markers_shown(self):
        sv = SeatsVotesPlot()
        assert sv._display_election_markers is True

    def test_default_linewidth(self):
        sv = SeatsVotesPlot()
        assert sv.linewidth == 2.5

    def test_default_markersize(self):
        sv = SeatsVotesPlot()
        assert sv.markersize == 8.0


# =========================
# == ADD SEAT VOTES DATA ==
# =========================


class TestSeatsVotesCrosshairs:
    def test_set_crosshair_options(self):
        sv = SeatsVotesPlot()
        sv.set_crosshair_options(x_width=0.05, y_width=0.05)
        style = sv._crosshair_style
        assert style is not None
        assert style.x_width == pytest.approx(0.05)
        assert style.y_width == pytest.approx(0.05)

    def test_remove_crosshairs(self):
        sv = SeatsVotesPlot()
        sv.remove_crosshairs()
        assert sv._crosshair_style is None


# ====================================
# == MARKER/LINE VISIBILITY TOGGLES ==
# ====================================


class TestSeatsVotesVisibility:
    def test_hide_election_markers(self):
        sv = SeatsVotesPlot()
        sv.display_election_markers(False)
        assert sv._display_election_markers is False

    def test_show_election_markers(self):
        sv = SeatsVotesPlot()
        sv.display_election_markers(False)
        sv.display_election_markers(True)
        assert sv._display_election_markers is True

    def test_hide_additional_lines_in_legend(self):
        sv = SeatsVotesPlot()
        sv.display_additional_lines_in_legend(False)
        assert sv._display_line_legend is False

    def test_show_additional_lines_in_legend(self):
        sv = SeatsVotesPlot()
        sv.display_additional_lines_in_legend(False)
        sv.display_additional_lines_in_legend(True)
        assert sv._display_line_legend is True


# ================================
# == LEGEND HANDLES COMPOSITION ==
# ================================
