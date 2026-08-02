"""Tests for PaintballPlot edge cases."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from gerrytools.plotting.data.paintball import PaintballPlot
from tests.plotting.paintball._helpers import simple_paintball


# ====================================
# == DEGENERATE HULL (< 3 VERTICES) ==
# ====================================
class TestPaintballHullPolygonDegenerate:
    """Degenerate hulls are rendered without polygon errors."""

    def test_two_identical_points_create_degenerate_hull(self):
        """Two identical data points give a degenerate horizontal hull."""
        plot = PaintballPlot()
        plot.add_seats_votes_data([0.5, 0.5], [0.5, 0.5])
        plot.display_hull(True)
        ax = plot.ax
        assert ax is not None

    def test_single_unique_y_two_x_gives_two_hull_vertices(self):
        """Points sharing a seat share produce a two-vertex horizontal hull."""
        plot = PaintballPlot()
        plot.add_seats_votes_data([0.4, 0.6], [0.5, 0.5])
        plot.display_hull(True)
        ax = plot.ax
        assert ax is not None


# ================================
# == LINE LEGEND: LABEL IS NONE ==
# ================================
class TestPaintballLineLegend:
    """Unnamed lines are omitted from legend handles."""

    def test_unnamed_line_not_in_legend_handles(self):
        """Anonymous slope lines do not change the legend."""
        plot = PaintballPlot(legend=True)
        plot.add_seats_votes_data([0.5], [0.5])
        baseline_labels = [handle.get_label() for handle in plot._legend_handles]
        plot.add_lines_with_slope(slopes=[1.5], linecolor="red", name=None)
        labels = [handle.get_label() for handle in plot._legend_handles]

        assert baseline_labels == ["Plan Outcomes"]
        assert labels == baseline_labels


# =================
# == SHOW METHOD ==
# =================
class TestPaintballShow:
    """The non-GUI show path writes an image file."""

    def test_show_saves_file_in_agg_backend(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        plot = simple_paintball()
        plot.show()
        saved = Path(capsys.readouterr().out.strip().rsplit("saved to ", 1)[1])
        assert saved.exists()
        saved.unlink()

    def test_show_with_hull_saves_file(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        plot = simple_paintball()
        plot.display_hull(True)
        plot.show()
        saved = Path(capsys.readouterr().out.strip().rsplit("saved to ", 1)[1])
        assert saved.exists()
        saved.unlink()


# =================
# == SAVE METHOD ==
# =================
class TestPaintballSave:
    """The save path writes the expected files."""

    def test_save_creates_file(self, tmp_path):
        plot = simple_paintball()
        out_path = str(tmp_path / "test.png")
        plot.save(out_path)
        assert (tmp_path / "test.png").exists()

    def test_save_with_hull_creates_file(self, tmp_path):
        plot = simple_paintball()
        plot.display_hull(True)
        out_path = str(tmp_path / "test_hull.png")
        plot.save(out_path)
        assert (tmp_path / "test_hull.png").exists()


# =========================
# == UNNAMED SLOPE LINES ==
# =========================


class TestPaintballUnnamedLines:
    """Unnamed slope lines still render correctly."""

    def test_unnamed_lines_rendered_without_error(self):
        """add_lines_with_slope without name= stores in self._lines and renders."""
        plot = PaintballPlot(legend=False)
        plot.add_seats_votes_data([0.45, 0.50, 0.55], [0.4, 0.5, 0.6])
        # No name → goes into self._lines
        plot.add_lines_with_slope(slopes=[1.0, 2.0])
        ax = plot.ax
        assert ax is not None


# ================
# == EMPTY DATA ==
# ================
class TestEmptyPaintballBuild:
    """Building with no data raises in both draw modes (hull mode used to crash on zip)."""

    def test_points_mode_raises_valueerror(self):
        plot = PaintballPlot()
        with pytest.raises(ValueError, match="No paintball data added yet"):
            plot.ax

    def test_hull_mode_raises_valueerror(self):
        plot = PaintballPlot()
        plot.display_hull(True)
        with pytest.raises(ValueError, match="No paintball data added yet"):
            plot.ax
