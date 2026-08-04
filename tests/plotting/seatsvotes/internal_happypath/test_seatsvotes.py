import numpy as np
import pytest
from matplotlib.lines import Line2D

from gerrytools.plotting.data.seatsvotes import SeatsVotesPlot


def test_seatsvotes_supports_per_series_line_and_marker_options():
    plot = SeatsVotesPlot(legend=True)
    plot.add_election(
        target_party_vote_shares=[0.46, 0.52, 0.57],
        name="Election A",
        linecolor="denim",
        linealpha=0.8,
        linestyle="--",
        linewidth=3.5,
        zorder=3,
        markerfacecolor="alizarin",
        markerfacealpha=0.6,
        marker="s",
        markersize=9.0,
        markeredgecolor="black",
        markeredgealpha=1.0,
        markeredgewidth=1.2,
        marker_zorder=4,
        marker_label="Result A",
    )
    plot.add_proportionality_line(
        linecolor="grey", linealpha=0.5, linestyle=":", linewidth=1.5, zorder=-2, name="Prop"
    )
    plot.add_efficiency_gap_line(
        linecolor="black", linealpha=0.4, linestyle="-", linewidth=1.0, zorder=-3, name="EG"
    )
    plot.add_custom_line(
        slope=0.5,
        linecolor="green",
        linealpha=0.7,
        linestyle="-.",
        linewidth=2.0,
        zorder=-4,
        name="Custom",
    )

    ax = plot.ax
    assert len(ax.lines) >= 5

    handles = plot._legend_handles
    labels = {h.get_label() for h in handles}
    assert {"Election A", "Result A", "Prop", "EG", "Custom"}.issubset(labels)

    marker_handle = next(h for h in handles if h.get_label() == "Result A")
    assert isinstance(marker_handle, Line2D)
    assert marker_handle.get_marker() == "s"
    assert marker_handle.get_markersize() == pytest.approx(9.0)
    assert marker_handle.get_markeredgewidth() == pytest.approx(1.2)

    curve_handle = next(h for h in handles if h.get_label() == "Election A")
    assert isinstance(curve_handle, Line2D)
    assert curve_handle.get_linestyle() == "--"
    assert curve_handle.get_linewidth() == pytest.approx(3.5)


def test_seatsvotes_mismatched_total_votes_length_raises():
    plot = SeatsVotesPlot()
    with pytest.raises(ValueError, match="must match per district"):
        plot.add_election([100, 200, 300], [400, 500])


def test_seatsvotes_extreme_curve_x_values_are_monotonic():
    plot = SeatsVotesPlot()
    plot.add_election([0.99, 0.01, 0.01])

    curve = plot.ax.lines[0]

    assert np.all(np.diff(curve.get_xdata()) >= 0)


class TestSeatsVotesDataEdgeCases:
    def test_infinite_markersize_raises_valueerror(self):
        with pytest.raises(ValueError, match="markersize must be finite"):
            plot = SeatsVotesPlot()
            plot.add_election([0.4, 0.6], markersize=float("inf"))

    def test_infinite_markeredgewidth_raises_valueerror(self):
        with pytest.raises(ValueError, match="markeredgewidth must be finite"):
            plot = SeatsVotesPlot()
            plot.add_election([0.4, 0.6], markeredgewidth=float("inf"))
