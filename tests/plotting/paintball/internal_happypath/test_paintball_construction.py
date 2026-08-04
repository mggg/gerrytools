import math

import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data._unit_square_base import _SlopeLine
from gerrytools.plotting.data.paintball import PaintballPlot
from tests.plotting._typing_utils import as_any
from tests.plotting.paintball._helpers import simple_paintball


# =============================
# == PAINTBALLLINE DATACLASS ==
# =============================
class TestSlopeLineDataclass:
    def test_valid_construction(self):
        line = _SlopeLine(slope=2.0, linecolor="black", linewidth=1.0, linestyle="-")
        assert line.slope == 2.0
        assert line.linewidth == 1.0

    def test_nan_slope_raises_valueerror(self):
        with pytest.raises(ValueError, match="NaN"):
            _SlopeLine(slope=float("nan"), linecolor="black", linewidth=1.0, linestyle="-")

    def test_infinite_slope_is_allowed(self):
        line = _SlopeLine(slope=float("inf"), linecolor="black", linewidth=1.0, linestyle="-")
        assert math.isinf(line.slope)

    def test_negative_slope_is_allowed(self):
        line = _SlopeLine(slope=-1.5, linecolor="black", linewidth=1.0, linestyle="-")
        assert line.slope == -1.5

    def test_zero_slope_is_allowed(self):
        line = _SlopeLine(slope=0.0, linecolor="black", linewidth=1.0, linestyle="-")
        assert line.slope == 0.0

    def test_infinite_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            _SlopeLine(slope=1.0, linecolor="black", linewidth=float("inf"), linestyle="-")

    def test_negative_linewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="nonnegative"):
            _SlopeLine(slope=1.0, linecolor="black", linewidth=-1.0, linestyle="-")

    def test_zero_linewidth_is_allowed(self):
        line = _SlopeLine(slope=1.0, linecolor="black", linewidth=0.0, linestyle="-")
        assert line.linewidth == 0.0

    def test_slope_coerced_to_float(self):
        line = _SlopeLine(slope=2, linecolor="black", linewidth=1.0, linestyle="-")
        assert isinstance(line.slope, float)

    def test_linewidth_coerced_to_float(self):
        line = _SlopeLine(slope=1.0, linecolor="black", linewidth=3, linestyle="-")
        assert isinstance(line.linewidth, float)

    def test_zorder_coerced_to_int(self):
        line = _SlopeLine(
            slope=1.0,
            linecolor="black",
            linewidth=1.0,
            linestyle="-",
            zorder=as_any(2.7),
        )
        assert isinstance(line.zorder, int)
        assert line.zorder == 2

    def test_default_label_is_none(self):
        line = _SlopeLine(slope=1.0, linecolor="black", linewidth=1.0, linestyle="-")
        assert line.label is None

    def test_frozen_prevents_mutation(self):
        line = _SlopeLine(slope=1.0, linecolor="black", linewidth=1.0, linestyle="-")
        with pytest.raises(AttributeError):
            setattr(line, "slope", 5.0)


# ======================================
# == CONSTRUCTION AND DATA VALIDATION ==
# ======================================


class TestPaintBallConstruction:
    def test_minimal_valid_construction(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.5], [0.5])
        assert len(pb._voteshare_data) == 1
        assert len(pb._seatshare_data) == 1

    def test_data_stored_as_floats(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([1, 0], [1, 0])
        assert all(isinstance(v, float) for v in pb._voteshare_data)
        assert all(isinstance(s, float) for s in pb._seatshare_data)

    def test_length_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="same length"):
            _pb = PaintballPlot()
            _pb.add_seats_votes_data([0.5], [0.5, 0.6])

    def test_empty_data_raises_valueerror(self):
        with pytest.raises(ValueError, match="at least one element"):
            _pb = PaintballPlot()
            _pb.add_seats_votes_data([], [])

    def test_voteshare_above_one_raises_valueerror(self):
        with pytest.raises(ValueError, match="vote-share values must be in"):
            _pb = PaintballPlot()
            _pb.add_seats_votes_data([1.1], [0.5])

    def test_voteshare_below_zero_raises_valueerror(self):
        with pytest.raises(ValueError, match="vote-share values must be in"):
            _pb = PaintballPlot()
            _pb.add_seats_votes_data([-0.1], [0.5])

    def test_seatshare_above_one_without_max_seats_raises_valueerror(self):
        with pytest.raises(ValueError, match="seat-share values must be in"):
            _pb = PaintballPlot()
            _pb.add_seats_votes_data([0.5], [1.5])

    def test_seatshare_below_zero_without_max_seats_raises_valueerror(self):
        with pytest.raises(ValueError, match="seat-share values must be in"):
            _pb = PaintballPlot()
            _pb.add_seats_votes_data([0.5], [-0.1])

    def test_maximum_seats_normalizes_seat_counts(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.5], [5], total_seats=10)
        assert pb._seatshare_data[0] == pytest.approx(0.5)

    def test_maximum_seats_zero_raises_valueerror(self):
        pb = PaintballPlot()
        with pytest.raises(ValueError, match="positive integer"):
            pb.add_seats_votes_data([0.5], [5], total_seats=0)

    def test_maximum_seats_negative_raises_valueerror(self):
        pb = PaintballPlot()
        with pytest.raises(ValueError, match="positive integer"):
            pb.add_seats_votes_data([0.5], [5], total_seats=-1)

    def test_seat_count_exceeds_max_seats_raises_valueerror(self):
        pb = PaintballPlot()
        with pytest.raises(ValueError, match="seat-share values must be in"):
            pb.add_seats_votes_data([0.5], [20], total_seats=10)

    def test_boundary_voteshare_zero_is_valid(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.0], [0.5])
        assert pb._voteshare_data[0] == 0.0

    def test_boundary_voteshare_one_is_valid(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([1.0], [0.5])
        assert pb._voteshare_data[0] == 1.0

    def test_boundary_seatshare_zero_is_valid(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.5], [0.0])
        assert pb._seatshare_data[0] == 0.0

    def test_boundary_seatshare_one_is_valid(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.5], [1.0])
        assert pb._seatshare_data[0] == 1.0


# ==================================
# == DEFAULT LINE INCLUSION FLAGS ==
# ==================================


class TestPaintBallDefaultLines:
    def test_default_includes_efficiency_gap_and_proportionality(self):
        pb = simple_paintball()
        assert "Efficiency Gap" in pb._named_lines
        assert "Proportionality" in pb._named_lines

    def test_efficiency_gap_line_has_slope_two(self):
        pb = simple_paintball()
        assert pb._named_lines["Efficiency Gap"].lines[0].slope == 2.0

    def test_proportionality_line_has_slope_one(self):
        pb = simple_paintball()
        assert pb._named_lines["Proportionality"].lines[0].slope == 1.0

    def test_disable_efficiency_gap_line(self):
        pb = simple_paintball(add_efficiency_gap_line=False)
        assert "Efficiency Gap" not in pb._named_lines

    def test_disable_proportionality_line(self):
        pb = simple_paintball(add_proportionality_line=False)
        assert "Proportionality" not in pb._named_lines

    def test_disable_both_default_lines(self):
        pb = simple_paintball(add_efficiency_gap_line=False, add_proportionality_line=False)
        assert len(pb._named_lines) == 0

    def test_default_legend_is_false(self):
        pb = simple_paintball()
        assert pb.legend is False

    def test_default_constructor_does_not_claim_legend_unit(self):
        pb = PaintballPlot()
        assert pb.legend is False
        assert not pb._axes_state.is_reclaimed("legend")

    def test_explicit_include_legend_false_claims_legend_unit(self):
        pb = PaintballPlot(legend=False)
        assert pb.legend is False
        assert pb._axes_state.is_reclaimed("legend")

    def test_explicit_include_legend_true_claims_legend_unit(self):
        pb = PaintballPlot(legend=True)
        assert pb.legend is True
        assert pb._axes_state.is_reclaimed("legend")

    def test_default_paintball_leaves_external_legend_alone(self):
        import matplotlib.pyplot as plt

        _, user_ax = plt.subplots()
        user_ax.plot([0, 1], [0, 1], label="external")
        user_ax.legend()
        external_legend = user_ax.get_legend()
        assert external_legend is not None

        pb = PaintballPlot(ax=user_ax)
        pb.add_seats_votes_data([0.4, 0.5, 0.6], [0.3, 0.5, 0.7])
        pb.ax
        assert pb._ax.get_legend() is external_legend


# ==================================
# == ADD VOTESHARE SEATSHARE DATA ==
# ==================================


class TestAddVoteshareSeatshareData:
    def test_adds_to_existing_data(self):
        pb = simple_paintball()
        original_length = len(pb._voteshare_data)
        pb.add_seats_votes_data([0.45], [0.55])
        assert len(pb._voteshare_data) == original_length + 1

    def test_new_data_appended_not_replaced(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.5], [0.5])
        pb.add_seats_votes_data([0.6], [0.7])
        assert pb._voteshare_data == [0.5, 0.6]
        assert pb._seatshare_data == [0.5, 0.7]

    def test_add_with_maximum_seats_normalization(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.5], [0.5])
        pb.add_seats_votes_data([0.6], [9], total_seats=18)
        assert pb._seatshare_data[-1] == pytest.approx(0.5)

    def test_add_invalid_data_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="vote-share values must be in"):
            pb.add_seats_votes_data([1.5], [0.5])

    def test_add_mismatched_length_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="same length"):
            pb.add_seats_votes_data([0.5, 0.6], [0.5])

    def test_add_empty_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="at least one element"):
            pb.add_seats_votes_data([], [])


# =====================
# == LINE MANAGEMENT ==
# =====================
