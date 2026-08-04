"""Tests for the shared electoral math and its cross-backend consistency.

The latex (TikZ) and plotting (Matplotlib) backends must produce identical coordinates and
curve values from identical inputs; the shared ``gerrytools._election_math`` functions are the
single source of that math.
"""

import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from gerrytools._election_math import (
    horizontal_hull_vertices,
    normalize_paintball_data,
    overall_election_point,
    paintball_coordinates,
    seats_votes_curve_values,
)

# Deliberately asymmetric shares make swapped or complemented coordinate conventions visible.
ASYMMETRIC_VOTESHARES = [0.31, 0.62, 0.62, 0.9]
ASYMMETRIC_SEATSHARES = [0.25, 0.5, 0.75, 0.75]

TIKZ_POINT_RE = re.compile(r"(-?\d+\.\d+)/(-?\d+\.\d+)")
TIKZ_COORD_RE = re.compile(r"\((-?\d+\.\d+),(-?\d+\.\d+)\)")


def test_importing_election_math_does_not_import_the_scoring_stack():
    # Regression: importing gerrytools.scoring.formulas executed the scoring package __init__,
    # pulling networkx, geopandas, and the scoring engine into this low-level helper.
    check = (
        "import sys; import gerrytools._election_math; "
        "assert not any(m.startswith('networkx') or m.startswith('geopandas') "
        "for m in sys.modules), "
        "[m for m in sys.modules if m.startswith(('networkx','geopandas'))]"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


class TestSharedFunctions:
    def test_seats_votes_curve_values(self):
        pov_counts = np.array([30.0, 60.0])
        total_counts = np.array([100.0, 100.0])

        vote_breaks, seat_breaks = seats_votes_curve_values(pov_counts, total_counts)

        # Overall share is 0.45; breakpoints are 0.45 - share + 0.5 for each district.
        assert vote_breaks == pytest.approx([0.0, 0.35, 0.65, 1.0])
        assert seat_breaks == pytest.approx([0.0, 0.0, 0.5, 1.0])

    def test_seats_votes_curve_values_clip_extreme_breakpoints(self):
        vote_breaks, _ = seats_votes_curve_values(
            np.array([99.0, 1.0, 1.0]),
            np.array([100.0, 100.0, 100.0]),
        )

        assert vote_breaks == sorted(vote_breaks)
        assert vote_breaks[0] == 0.0
        assert vote_breaks[-1] == 1.0

    def test_seats_votes_curve_values_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="must have the same shape"):
            seats_votes_curve_values(np.array([1.0]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="must be positive"):
            seats_votes_curve_values(np.array([1.0]), np.array([0.0]))

    def test_seats_votes_curve_values_rejects_pov_exceeding_total(self):
        # Regression: the error used to surface the internal name "opposition_votes".
        with pytest.raises(ValueError, match="pov_counts cannot exceed total_counts"):
            seats_votes_curve_values(np.array([150.0]), np.array([100.0]))

    def test_overall_election_point(self):
        pov_counts = np.array([30.0, 60.0, 80.0])
        total_counts = np.array([100.0, 100.0, 100.0])

        vote_share, seat_share = overall_election_point(pov_counts, total_counts)

        assert vote_share == pytest.approx(170.0 / 300.0)
        assert seat_share == pytest.approx(2.0 / 3.0)

    def test_overall_election_point_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="must have the same shape"):
            overall_election_point(np.array([1.0]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="must be positive"):
            overall_election_point(np.array([1.0]), np.array([0.0]))
        with pytest.raises(ValueError, match="pov_counts cannot exceed total_counts"):
            overall_election_point(np.array([150.0]), np.array([100.0]))

    def test_normalize_paintball_data_scales_seat_counts(self):
        voteshares, seatshares = normalize_paintball_data([0.4, 0.6], [2, 3], 4)
        assert voteshares == [0.4, 0.6]
        assert seatshares == [0.5, 0.75]

    @pytest.mark.parametrize(
        "voteshares,seats,total_seats,message",
        [
            ([0.5], [0.5, 0.6], None, "same length"),
            ([], [], None, "at least one element"),
            ([1.5], [0.5], None, "vote-share values must be in"),
            ([0.5], [1.5], None, "seat-share values must be in"),
            ([0.5], [8], 4, "seat-share values must be in"),
            ([0.5], [1], 0, "total_seats must be a positive integer"),
            ([0.5], [1], 4.5, "total_seats must be a positive integer"),
            ([0.5], [1], True, "total_seats must be a positive integer"),
        ],
    )
    def test_normalize_paintball_data_rejects_bad_inputs(
        self, voteshares, seats, total_seats, message
    ):
        with pytest.raises(ValueError, match=message):
            normalize_paintball_data(voteshares, seats, total_seats)

    def test_paintball_coordinates_are_direct_votes_seats_points(self):
        x_coords, y_coords = paintball_coordinates([0.31, 0.9], [0.25, 0.75])
        assert x_coords == pytest.approx([0.31, 0.9])
        assert y_coords == pytest.approx([0.25, 0.75])

    def test_horizontal_hull_tracks_min_and_max_x_per_y(self):
        vertices = horizontal_hull_vertices([(0.5, 0.5), (0.8, 0.5), (0.2, 0.5), (0.4, 0.9)])
        assert vertices == [(0.2, 0.5), (0.4, 0.9), (0.4, 0.9), (0.8, 0.5)]

    def test_horizontal_hull_of_no_points_is_empty(self):
        assert horizontal_hull_vertices([]) == []

    def test_horizontal_hull_of_single_point_repeats_it_for_both_sides(self):
        # One point yields a degenerate polygon: the point as both its left and right side.
        assert horizontal_hull_vertices([(0.3, 0.4)]) == [(0.3, 0.4), (0.3, 0.4)]


class TestCrossBackendSeatsVotes:
    def test_curve_values_and_marker_identical_across_backends(self):
        from gerrytools.latex.seatsvotes import _SeatsVotesData as LatexData
        from gerrytools.plotting.data.seatsvotes import _SeatsVotesData as MplData

        pov_counts = np.array([310.0, 620.0, 900.0])
        total_counts = np.array([1000.0, 1000.0, 1000.0])

        latex_series = LatexData(
            pov_party_vote_counts=pov_counts,
            total_vote_counts=total_counts,
            name="series",
            linecolor="black",
            markercolor="black",
            marker_label="marker",
        )
        from gerrytools.plotting.data.options import (
            SeatsVotesLineOptions,
            SeatsVotesMarkerOptions,
        )

        mpl_series = MplData(
            pov_party_vote_counts=pov_counts,
            total_vote_counts=total_counts,
            name="series",
            line_style=SeatsVotesLineOptions(linecolor="black"),
            marker_style=SeatsVotesMarkerOptions(markerfacecolor="black"),
            marker_label="marker",
        )

        assert latex_series.seats_votes_curve_values() == mpl_series.seats_votes_curve_values()


class TestCrossBackendPaintball:
    def _latex_plot(self):
        from gerrytools.latex.paintball import TikzPaintballPlot as LatexPaintballPlot

        return LatexPaintballPlot(
            vote_share_data=ASYMMETRIC_VOTESHARES,
            seats_data=ASYMMETRIC_SEATSHARES,
        )

    def _mpl_plot(self):
        from gerrytools.plotting.data.paintball import PaintballPlot as MplPaintballPlot

        plot = MplPaintballPlot()
        plot.add_seats_votes_data(ASYMMETRIC_VOTESHARES, ASYMMETRIC_SEATSHARES)
        return plot

    def test_point_coordinates_identical_across_backends(self):
        # Both backends must use direct (vote share, seat share) coordinates.
        latex_points = [
            (float(x), float(y))
            for x, y in TIKZ_POINT_RE.findall(self._latex_plot()._paintball_points_str())
        ]

        mpl_xs, mpl_ys = self._mpl_plot()._paintball_coordinates()
        mpl_points = [(round(x, 4), round(y, 4)) for x, y in zip(mpl_xs, mpl_ys)]

        assert latex_points == pytest.approx(mpl_points)

    def test_hull_vertices_identical_across_backends(self):
        latex_hull = [
            (float(x), float(y))
            for x, y in TIKZ_COORD_RE.findall(self._latex_plot()._paintball_hull_str())
        ]

        mpl_hull = [
            (round(x, 4), round(y, 4)) for x, y in self._mpl_plot()._horizontal_hull_vertices()
        ]

        assert latex_hull == pytest.approx(mpl_hull)

    def test_latex_hull_bounds_the_emitted_points(self):
        # Every emitted point must lie within the horizontal hull for its seat share.
        plot = self._latex_plot()
        points = [
            (float(x), float(y)) for x, y in TIKZ_POINT_RE.findall(plot._paintball_points_str())
        ]
        hull = [(float(x), float(y)) for x, y in TIKZ_COORD_RE.findall(plot._paintball_hull_str())]

        hull_x_by_y: dict[float, list[float]] = {}
        for x, y in hull:
            hull_x_by_y.setdefault(y, []).append(x)

        assert points
        for x, y in points:
            assert y in hull_x_by_y
            assert min(hull_x_by_y[y]) <= x <= max(hull_x_by_y[y])
