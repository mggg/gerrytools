import matplotlib

matplotlib.use("Agg")

from tests.plotting.paintball._helpers import simple_paintball


# =============================
# == PAINTBALLLINE DATACLASS ==
# =============================
class TestPaintBallBuild:
    def test_display_hull_is_persistent_and_reversible(self):
        pb = simple_paintball()
        assert pb._draw_hull is False
        pb.display_hull(True)
        assert pb._draw_hull is True
        pb.display_hull(False)
        assert pb._draw_hull is False

    def test_clear_options_resets_hull_mode(self):
        pb = simple_paintball()
        pb.display_hull(True)
        pb.clear_options()
        assert pb._draw_hull is False

    def test_build_with_no_default_lines(self):
        pb = simple_paintball(
            add_efficiency_gap_line=False,
            add_proportionality_line=False,
        )
        assert all(line.get_linestyle() == "None" for line in pb.ax.lines)


# ====================
# == LEGEND HANDLES ==
# ====================


class TestPaintBallLegendHandles:
    def test_point_view_legend_has_plan_outcomes(self):
        pb = simple_paintball()
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Plan Outcomes" in labels

    def test_point_view_legend_has_named_lines(self):
        pb = simple_paintball()
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Efficiency Gap" in labels
        assert "Proportionality" in labels

    def test_hull_view_legend_has_horizontal_hull(self):
        pb = simple_paintball()
        pb._draw_hull = True
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Horizontal Hull" in labels

    def test_hull_view_legend_excludes_plan_outcomes(self):
        pb = simple_paintball()
        pb._draw_hull = True
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "Plan Outcomes" not in labels

    def test_no_named_lines_means_fewer_legend_handles(self):
        pb = simple_paintball(
            add_efficiency_gap_line=False,
            add_proportionality_line=False,
        )
        handles = pb._legend_handles
        # Only the Plan Outcomes handle
        assert len(handles) == 1
        assert handles[0].get_label() == "Plan Outcomes"

    def test_anonymous_lines_excluded_from_legend(self):
        pb = simple_paintball(
            add_efficiency_gap_line=False,
            add_proportionality_line=False,
        )
        pb.add_lines_with_slope(slopes=[3.0])  # no name
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        # Only Plan Outcomes, no label for the anonymous line
        assert len(labels) == 1

    def test_custom_named_line_appears_in_legend(self):
        pb = simple_paintball(
            add_efficiency_gap_line=False,
            add_proportionality_line=False,
        )
        pb.add_lines_with_slope(slopes=[1.5], name="My Guide")
        handles = pb._legend_handles
        labels = [h.get_label() for h in handles]
        assert "My Guide" in labels


# ==============================
# == TICK LABEL RESTORE FLOW  ==
# ==============================


class TestPaintballTickLabelRestore:
    """Restoring tick labels after a render shows them again.

    Regression test: ``clear_options`` hides both axes' labels via empty ticks, and the
    ordinary render -> ``set_xticks(..., labels=...)`` -> render flow must re-enable label
    visibility, not just write invisible label text.
    """

    def test_set_ticks_after_render_shows_labels(self):
        pb = simple_paintball()
        _ = pb.ax  # first render applies the hidden-label default
        pb.set_xticks([0.0, 0.5, 1.0], labels=["0", "1/2", "1"])
        pb.set_yticks([0.0, 0.5, 1.0], labels=["0", "1/2", "1"])
        ax = pb.ax
        assert [tick.get_text() for tick in ax.get_xticklabels()] == ["0", "1/2", "1"]
        assert all(tick.get_visible() for tick in ax.get_xticklabels())
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["0", "1/2", "1"]
        assert all(tick.get_visible() for tick in ax.get_yticklabels())
