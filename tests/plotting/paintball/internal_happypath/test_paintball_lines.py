import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data.paintball import PaintballPlot
from tests.plotting.paintball._helpers import simple_paintball


# =============================
# == PAINTBALLLINE DATACLASS ==
# =============================
class TestPaintBallLineManagement:
    def test_add_named_line(self):
        pb = simple_paintball(add_efficiency_gap_line=False, add_proportionality_line=False)
        pb.add_lines_with_slope(slopes=[1.5], name="Custom Line")
        assert "Custom Line" in pb._named_lines
        assert pb._named_lines["Custom Line"].lines[0].slope == 1.5

    def test_add_anonymous_line(self):
        pb = simple_paintball(add_efficiency_gap_line=False, add_proportionality_line=False)
        pb.add_lines_with_slope(slopes=[3.0], linecolor="red")
        assert [line.slope for line in pb._lines] == [3.0]

    def test_add_multiple_slopes_at_once(self):
        pb = simple_paintball(add_efficiency_gap_line=False, add_proportionality_line=False)
        pb.add_lines_with_slope(slopes=[1.0, 2.0, 3.0], linecolor="blue")
        assert [line.slope for line in pb._lines] == [1.0, 2.0, 3.0]

    def test_add_multiple_slopes_with_name_stores_and_draws_one_group(self):
        pb = simple_paintball(add_efficiency_gap_line=False, add_proportionality_line=False)
        pb.add_lines_with_slope(slopes=[1.0, 2.0], name="Shared Name")

        assert [line.slope for line in pb._named_lines["Shared Name"]] == [1.0, 2.0]
        assert sum(line.get_linestyle() == "-" for line in pb.ax.lines) == 2
        assert [handle.get_label() for handle in pb._slope_line_legend_handles()] == ["Shared Name"]

    def test_second_named_slope_group_replaces_the_first(self):
        pb = simple_paintball(add_efficiency_gap_line=False, add_proportionality_line=False)
        pb.add_lines_with_slope(slopes=[1.0, 2.0], name="Shared Name")
        pb.add_lines_with_slope(slopes=[3.0], name="Shared Name")

        assert [line.slope for line in pb._named_lines["Shared Name"]] == [3.0]

    def test_add_line_with_custom_properties(self):
        pb = simple_paintball(add_efficiency_gap_line=False, add_proportionality_line=False)
        pb.add_lines_with_slope(
            slopes=[1.0],
            linecolor="red",
            linewidth=2.5,
            linestyle="--",
            linealpha=0.7,
            zorder=5,
            name="Styled",
        )
        line = pb._named_lines["Styled"].lines[0]
        assert line.linewidth == 2.5
        assert line.linestyle == "--"
        assert line.zorder == 5

    def test_clear_lines_removes_all(self):
        pb = simple_paintball()
        pb.add_lines_with_slope(slopes=[3.0])
        pb.clear_lines()
        assert len(pb._named_lines) == 0
        assert len(pb._lines) == 0

    def test_clear_lines_leaves_data_intact(self):
        pb = simple_paintball()
        original_data_len = len(pb._voteshare_data)
        pb.clear_lines()
        assert len(pb._voteshare_data) == original_data_len


# =========================
# == SET XLIM / SET YLIM ==
# =========================


class TestPaintBallCoordinates:
    def test_coordinates_are_direct_votes_seats_points(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.3, 0.7], [0.2, 0.8])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(0.3), pytest.approx(0.7)]
        assert ys == [pytest.approx(0.2), pytest.approx(0.8)]

    def test_center_point_maps_to_center(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.5], [0.5])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(0.5)]
        assert ys == [pytest.approx(0.5)]

    def test_boundary_zero_stays_zero(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.0], [0.0])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(0.0)]
        assert ys == [pytest.approx(0.0)]

    def test_boundary_one_stays_one(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([1.0], [1.0])
        xs, ys = pb._paintball_coordinates()
        assert xs == [pytest.approx(1.0)]
        assert ys == [pytest.approx(1.0)]


# ==============================
# == HORIZONTAL HULL VERTICES ==
# ==============================


class TestHorizontalHullVertices:
    def test_single_point_hull(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.5], [0.5])
        vertices = pb._horizontal_hull_vertices()
        # Single point: one y level, left_side + right_side = 2 vertices
        assert len(vertices) == 2
        # Both vertices should be the same point
        assert vertices[0] == vertices[1]

    def test_two_points_same_y(self):
        # Both have the same seat share.
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.3, 0.7], [0.5, 0.5])
        vertices = pb._horizontal_hull_vertices()
        assert len(vertices) == 2
        xs = [vertex[0] for vertex in vertices]
        assert xs == [pytest.approx(0.3), pytest.approx(0.7)]

    def test_three_points_different_y_produces_hull(self):
        pb = PaintballPlot()
        pb.add_seats_votes_data([0.3, 0.5, 0.7], [0.2, 0.5, 0.8])
        vertices = pb._horizontal_hull_vertices()
        # 3 unique y-levels -> left_side(3) + right_side(3) = 6 vertices
        assert len(vertices) == 6

    def test_hull_vertices_are_tuples_of_floats(self):
        pb = simple_paintball()
        vertices = pb._horizontal_hull_vertices()
        for v in vertices:
            assert isinstance(v, tuple)
            assert len(v) == 2


# =========================
# == BUILD AND RENDERING ==
# =========================
