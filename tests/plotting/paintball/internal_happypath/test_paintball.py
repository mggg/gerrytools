import pytest

from gerrytools.plotting.data.paintball import PaintballPlot


def test_paintball_builds_point_and_hull_views():
    plot = PaintballPlot(legend=True)
    plot.add_seats_votes_data(
        [0.4925, 0.5233, 0.4960, 0.5259],
        [5, 10, 9, 9],
        total_seats=18,
    )
    plot.set_marker_options(
        size=9.0,
        color="alizarin",
        alpha=0.4,
        edgecolor="black",
        edgewidth=1.0,
        edgealpha=0.8,
        marker="s",
    )
    plot.set_hull_options(
        color="teagreen",
        alpha=0.5,
        edgecolor="green",
        edgewidth=1.2,
        edgealpha=0.9,
    )
    plot.add_lines_with_slope(slopes=[0.75], linealpha=0.5, zorder=-5, name="Custom")

    ax = plot.ax
    # Crosshairs are now span patches (matching SeatsVotesPlot); guide line + points are lines.
    assert len(ax.lines) >= 2
    assert len(ax.patches) >= 2
    assert any(line.get_marker() == "s" for line in ax.lines)
    assert any(line.get_zorder() == -5 for line in ax.lines)

    plot.display_hull(True)
    hull_ax = plot.ax
    assert len(hull_ax.patches) >= 1


def test_paintball_validates_data_shapes_and_ranges():
    with pytest.raises(ValueError, match="same length"):
        _pb = PaintballPlot()
        _pb.add_seats_votes_data([0.5], [0.5, 0.6])
    with pytest.raises(ValueError, match="vote-share values must be in \\[0, 1\\]"):
        _pb = PaintballPlot()
        _pb.add_seats_votes_data([1.2], [0.5])
    with pytest.raises(ValueError, match="total_seats must be a positive integer"):
        _pb = PaintballPlot()
        _pb.add_seats_votes_data([0.5], [2], total_seats=0)


def test_paintball_crosshair_options_match_seatsvotes_semantics():
    plot = PaintballPlot()
    plot.add_seats_votes_data([0.5], [0.5])

    plot.set_crosshair_options(color="black")
    style = plot._crosshair_style
    assert style is not None
    assert style.color == "black"
    assert style.x_width == 0.007
    assert style.alpha == 1.0
