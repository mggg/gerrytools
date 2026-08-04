import matplotlib

matplotlib.use("Agg")

import pytest

from tests.plotting.paintball._helpers import simple_paintball


# =============================
# == PAINTBALLLINE DATACLASS ==
# =============================
class TestPaintBallAxisLimits:
    def test_set_xlim_valid(self):
        pb = simple_paintball()
        pb.set_xlim(0.2, 0.8)
        assert pb._xaxis.limits == (0.2, 0.8)

    def test_set_ylim_valid(self):
        pb = simple_paintball()
        pb.set_ylim(0.1, 0.9)
        assert pb._yaxis.limits == (0.1, 0.9)


# ================
# == SET ASPECT ==
# ================


class TestPaintBallAspect:
    def test_set_aspect_valid(self):
        pb = simple_paintball()
        pb.set_aspect(1.5)
        assert pb._aspect_ratio == 1.5

    def test_default_aspect_is_square(self):
        pb = simple_paintball()
        assert pb._aspect_ratio == 1.0

    def test_set_aspect_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_aspect(0.0)

    def test_set_aspect_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="positive"):
            pb.set_aspect(-1.0)

    def test_set_aspect_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_aspect(float("inf"))

    def test_set_aspect_nan_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_aspect(float("nan"))

    def test_mpl_scale_names_are_not_shadowed(self):
        # The old set_xscale/set_yscale collided with Axes.set_xscale("log"); they are gone.
        pb = simple_paintball()
        assert not hasattr(pb, "set_xscale")
        assert not hasattr(pb, "set_yscale")
        assert not hasattr(pb, "set_scale")

    def test_aspect_applied_at_build(self):
        pb = simple_paintball()
        pb.set_aspect(2.0)
        ax = pb.ax
        assert float(ax.get_aspect()) == 2.0

    def test_set_aspect_reclaims_after_external_change(self):
        pb = simple_paintball()
        ax = pb.ax
        ax.set_aspect("auto")
        pb.title = "external aspect"
        pb.ax

        pb.set_aspect(2.0)

        assert float(pb.ax.get_aspect()) == 2.0


# ===========================
# == SET CROSSHAIR OPTIONS ==
# ===========================


class TestPaintBallCrosshairOptions:
    def test_set_options_replaces_style(self):
        pb = simple_paintball()
        pb.set_crosshair_options(color="blue", x_width=0.05, y_width=0.03, alpha=0.3)
        style = pb._crosshair_style
        assert style is not None
        assert style.color == "blue"
        assert style.x_width == 0.05
        assert style.y_width == 0.03
        assert style.alpha == 0.3

    def test_defaults_match_signature(self):
        pb = simple_paintball()
        pb.set_crosshair_options(color="red")
        style = pb._crosshair_style
        assert style is not None
        assert style.color == "red"
        assert style.x_width == 0.007

    def test_remove_crosshairs(self):
        pb = simple_paintball()
        pb.remove_crosshairs()
        assert pb._crosshair_style is None

    def test_width_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_crosshair_options(x_width=float("inf"))

    def test_width_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="nonnegative"):
            pb.set_crosshair_options(x_width=-1.0)

    def test_width_zero_is_valid(self):
        pb = simple_paintball()
        pb.set_crosshair_options(x_width=0.0)
        style = pb._crosshair_style
        assert style is not None
        assert style.x_width == 0.0

    def test_alpha_above_one_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_crosshair_options(alpha=1.5)

    def test_alpha_below_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_crosshair_options(alpha=-0.1)

    def test_alpha_boundaries_are_valid(self):
        pb = simple_paintball()
        pb.set_crosshair_options(alpha=0.0)
        pb.set_crosshair_options(alpha=1.0)


# ========================
# == SET MARKER OPTIONS ==
# ========================


class TestPaintBallMarkerOptions:
    def test_set_size(self):
        pb = simple_paintball()
        pb.set_marker_options(size=20.0)
        assert pb._marker_options.markersize == 20.0

    def test_set_color(self):
        pb = simple_paintball()
        pb.set_marker_options(color="red")
        assert pb._marker_options.markerfacecolor == "#ff0000"

    def test_set_none_colors_removes_marker_fill_and_edge(self):
        pb = simple_paintball()
        pb.set_marker_options(color=None, edgecolor=None)

        assert pb._marker_options.markerfacecolor == "none"
        assert pb._marker_options.markerfacealpha == 0.0
        assert pb._marker_options.markeredgecolor == "none"
        assert pb._marker_options.markeredgealpha == 0.0
        assert pb._marker_options.markeredgewidth == 0.0

    def test_set_marker_string(self):
        pb = simple_paintball()
        pb.set_marker_options(marker="s")
        assert pb._marker_options.marker == "s"

    def test_set_alpha(self):
        pb = simple_paintball()
        pb.set_marker_options(alpha=0.5)
        assert pb._marker_options.markerfacealpha == 0.5

    def test_set_edgecolor(self):
        pb = simple_paintball()
        pb.set_marker_options(edgecolor="blue")
        assert pb._marker_options.markeredgecolor == "#0000ff"

    def test_visible_edgecolor_restores_default_width_after_none(self):
        pb = simple_paintball()
        pb.set_marker_options(edgecolor=None)
        pb.set_marker_options(edgecolor="blue")

        assert pb._marker_options.markeredgecolor == "#0000ff"
        assert pb._marker_options.markeredgewidth == 0.5

    def test_explicit_zero_edgewidth_stays_hidden_with_visible_color(self):
        pb = simple_paintball()
        pb.set_marker_options(edgecolor=None)
        pb.set_marker_options(edgecolor="blue", edgewidth=0.0)

        assert pb._marker_options.markeredgewidth == 0.0

    def test_set_edgewidth(self):
        pb = simple_paintball()
        pb.set_marker_options(edgewidth=2.0)
        assert pb._marker_options.markeredgewidth == 2.0

    def test_set_edgealpha(self):
        pb = simple_paintball()
        pb.set_marker_options(edgealpha=0.3)
        assert pb._marker_options.markeredgealpha == 0.3

    def test_size_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="nonnegative"):
            pb.set_marker_options(size=-1.0)

    def test_size_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_marker_options(size=float("inf"))

    def test_direct_attribute_bypass_is_closed(self):
        # The old bare `pb.markersize = -5` bypassed validation; the attrs are gone.
        pb = simple_paintball()
        assert not hasattr(pb, "markersize")

    def test_edgewidth_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="nonnegative"):
            pb.set_marker_options(edgewidth=-0.5)

    def test_edgewidth_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_marker_options(edgewidth=float("inf"))

    def test_edgewidth_zero_is_valid(self):
        pb = simple_paintball()
        pb.set_marker_options(edgewidth=0.0, edgecolor="none")
        assert pb._marker_options.markeredgewidth == 0.0

    def test_partial_update_preserves_other_fields(self):
        pb = simple_paintball()
        original_marker = pb._marker_options.marker
        original_alpha = pb._marker_options.markerfacealpha
        pb.set_marker_options(size=99.0)
        assert pb._marker_options.marker == original_marker
        assert pb._marker_options.markerfacealpha == original_alpha


# ======================
# == SET HULL OPTIONS ==
# ======================


class TestPaintBallHullOptions:
    def test_set_hull_color(self):
        pb = simple_paintball()
        pb.set_hull_options(color="green")
        assert pb._hull_style.facecolor == "#00ff00"

    def test_set_none_colors_removes_hull_fill_and_edge(self):
        pb = simple_paintball()
        pb.set_hull_options(color=None, edgecolor=None)

        assert pb._hull_style.facecolor == "none"
        assert pb._hull_style.facealpha == 0.0
        assert pb._hull_style.edgecolor == "none"
        assert pb._hull_style.edgealpha == 0.0
        assert pb._hull_style.edgewidth == 0.0

    def test_set_hull_alpha(self):
        pb = simple_paintball()
        pb.set_hull_options(alpha=0.4)
        assert pb._hull_style.facealpha == 0.4

    def test_set_hull_edgecolor(self):
        pb = simple_paintball()
        pb.set_hull_options(edgecolor="black")
        assert pb._hull_style.edgecolor == "#000000"

    def test_visible_hull_edgecolor_restores_default_width_after_none(self):
        pb = simple_paintball()
        pb.set_hull_options(edgecolor=None)
        pb.set_hull_options(edgecolor="black")

        assert pb._hull_style.edgecolor == "#000000"
        assert pb._hull_style.edgewidth == 2.0

    def test_explicit_zero_hull_edgewidth_stays_hidden_with_visible_color(self):
        pb = simple_paintball()
        pb.set_hull_options(edgecolor=None)
        pb.set_hull_options(edgecolor="black", edgewidth=0.0)

        assert pb._hull_style.edgewidth == 0.0

    def test_set_hull_edgewidth(self):
        pb = simple_paintball()
        pb.set_hull_options(edgewidth=3.0)
        assert pb._hull_style.edgewidth == 3.0

    def test_set_hull_edgealpha(self):
        pb = simple_paintball()
        pb.set_hull_options(edgealpha=0.7)
        assert pb._hull_style.edgealpha == 0.7

    def test_hull_alpha_above_one_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_hull_options(alpha=1.5)

    def test_hull_alpha_below_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="alpha must be in"):
            pb.set_hull_options(alpha=-0.1)

    def test_hull_edgewidth_negative_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="nonnegative"):
            pb.set_hull_options(edgewidth=-1.0)

    def test_hull_edgewidth_infinite_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="finite"):
            pb.set_hull_options(edgewidth=float("inf"))

    def test_hull_edgewidth_zero_is_valid(self):
        pb = simple_paintball()
        pb.set_hull_options(edgewidth=0.0)
        assert pb._hull_style.edgewidth == 0.0

    def test_hull_edgealpha_above_one_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="edgealpha must be in"):
            pb.set_hull_options(edgealpha=1.5)

    def test_hull_edgealpha_below_zero_raises_valueerror(self):
        pb = simple_paintball()
        with pytest.raises(ValueError, match="edgealpha must be in"):
            pb.set_hull_options(edgealpha=-0.1)

    def test_default_hull_colors_inherit_marker(self):
        pb = simple_paintball()
        assert pb._hull_style.facecolor is None
        assert pb._hull_style.edgecolor is None

    def test_partial_hull_update_preserves_other_fields(self):
        pb = simple_paintball()
        pb.set_hull_options(color="red", edgewidth=4.0)
        pb.set_hull_options(alpha=0.5)
        assert pb._hull_style.facecolor == "#ff0000"
        assert pb._hull_style.edgewidth == 4.0


# ===================
# == CLEAR OPTIONS ==
# ===================


class TestPaintBallClearOptions:
    def test_clear_options_resets_markersize(self):
        pb = simple_paintball()
        pb.set_marker_options(size=99.0)
        pb.clear_options()
        assert pb._marker_options.markersize == 16.0

    def test_clear_options_resets_marker_to_circle(self):
        pb = simple_paintball()
        pb.set_marker_options(marker="s")
        pb.clear_options()
        assert pb._marker_options.marker == "o"

    def test_clear_options_resets_crosshair_style(self):
        pb = simple_paintball()
        pb.set_crosshair_options(x_width=0.5)
        pb.clear_options()
        style = pb._crosshair_style
        assert style is not None
        assert style.x_width == 0.007

    def test_clear_options_resets_hull_color_to_none(self):
        pb = simple_paintball()
        pb.set_hull_options(color="red")
        pb.clear_options()
        assert pb._hull_style.facecolor is None

    def test_clear_options_resets_aspect(self):
        pb = simple_paintball()
        pb.set_aspect(2.5)
        pb.clear_options()
        assert pb._aspect_ratio == 1.0

    def test_clear_options_does_not_remove_data(self):
        pb = simple_paintball()
        original_len = len(pb._voteshare_data)
        pb.clear_options()
        assert len(pb._voteshare_data) == original_len


# ==========================================
# == PAINTBALL COORDINATES TRANSFORMATION ==
# ==========================================
