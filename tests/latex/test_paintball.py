"""Tests for non-visual LaTeX paintball generation."""

import pytest

from gerrytools.latex.paintball import PaintballOptions, TikzPaintballPlot, _PaintballLine


def named_lines(plot: TikzPaintballPlot) -> dict[str, _PaintballLine]:
    return {name: line for name, line in plot._lines if name is not None}


def unnamed_lines(plot: TikzPaintballPlot) -> list[_PaintballLine]:
    return [line for name, line in plot._lines if name is None]


# ============================
# == DATA CLASSES & OPTIONS ==
# ============================
class TestPaintballDataclasses:
    def test_paintball_line_accepts_valid_tikz_linestyle(self):
        line = _PaintballLine(slope=1.0, linecolor="black", linewidth=1.5, linestyle="dashed")
        assert line.linestyle == "dashed"

    def test_paintball_line_accepts_matplotlib_tokens(self):
        # Sibling consistency with the latex seats-votes plot: "--" maps to "dashed".
        line = _PaintballLine(slope=1.0, linecolor="black", linewidth=1.5, linestyle="--")
        assert line.linestyle == "dashed"

    def test_paintball_line_rejects_invalid_tikz_linestyle(self):
        with pytest.raises(ValueError, match="Invalid linestyle"):
            _PaintballLine(
                slope=1.0,
                linecolor="black",
                linewidth=1.0,
                linestyle="zigzag",
            )

    def test_paintball_line_rejects_invalid_numeric_values(self):
        with pytest.raises(ValueError, match="slope must not be NaN"):
            _PaintballLine(
                slope=float("nan"),
                linecolor="black",
                linewidth=1.0,
                linestyle="solid",
            )
        with pytest.raises(ValueError, match="linewidth must be nonnegative"):
            _PaintballLine(
                slope=1.0,
                linecolor="black",
                linewidth=-1.0,
                linestyle="solid",
            )

    def test_paintball_options_validate_numeric_ranges(self):
        options = PaintballOptions()

        with pytest.raises(ValueError, match="markersize must be positive"):
            options.markersize = 0

        with pytest.raises(ValueError, match=r"markeralpha must be in \[0, 1\]"):
            options.markeralpha = 2

        with pytest.raises(ValueError, match="xlim\\[0\\] must be less than xlim\\[1\\]"):
            options.xlim = (1.0, 0.0)

        with pytest.raises(AttributeError, match="Unknown PaintballOptions attribute"):
            options.unknown = 1

    def test_paintball_options_validate_additional_ranges_and_none_assignments(self):
        options = PaintballOptions()

        options.hullalpha = None
        options.hulledgewidth = None
        options.hulledgealpha = None

        assert options.hullalpha is None
        assert options.hulledgewidth is None
        assert options.hulledgealpha is None

        with pytest.raises(ValueError, match="markeredgewidth must be nonnegative"):
            options.markeredgewidth = -1

        with pytest.raises(ValueError, match=r"markeredgealpha must be in \[0, 1\]"):
            options.markeredgealpha = 2

        with pytest.raises(ValueError, match=r"hullalpha must be in \[0, 1\]"):
            options.hullalpha = 2

        with pytest.raises(ValueError, match="hulledgewidth must be nonnegative"):
            options.hulledgewidth = -1

        with pytest.raises(ValueError, match=r"hulledgealpha must be in \[0, 1\]"):
            options.hulledgealpha = 2

        with pytest.raises(ValueError, match="crosshair_width must be nonnegative"):
            options.crosshair_width = -1

        with pytest.raises(ValueError, match="xscale must be positive"):
            options.xscale = 0

        with pytest.raises(ValueError, match="yscale must be positive"):
            options.yscale = 0

        with pytest.raises(ValueError, match="ylim\\[0\\] must be less than ylim\\[1\\]"):
            options.ylim = (1.0, 0.0)


# =========================
# == CONSTRUCTION & DATA ==
# =========================
class TestPaintballConstruction:
    def test_initialization_normalizes_seat_counts(self):
        plot = TikzPaintballPlot(
            vote_share_data=[0.4, 0.6],
            seats_data=[2, 3],
            total_seats=4,
        )

        assert plot._voteshare_data == [0.4, 0.6]
        assert plot._seatshare_data == [0.5, 0.75]

    def test_initialization_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5, 0.6])

    def test_initialization_rejects_empty_inputs(self):
        with pytest.raises(ValueError, match="at least one element"):
            TikzPaintballPlot(vote_share_data=[], seats_data=[])

    def test_initialization_rejects_invalid_seatshares_without_maximum_seats(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            TikzPaintballPlot(vote_share_data=[0.5], seats_data=[1.5])

    def test_initialization_rejects_invalid_scaled_seatshares(self):
        with pytest.raises(ValueError, match="seat-share values must be in"):
            TikzPaintballPlot(vote_share_data=[0.5], seats_data=[8], total_seats=4)

    def test_add_voteshare_seatshare_data_extends_existing_data(self):
        plot = TikzPaintballPlot(vote_share_data=[0.4], seats_data=[0.5])
        plot.add_seats_votes_data([0.6], [0.75])

        assert plot._voteshare_data == [0.4, 0.6]
        assert plot._seatshare_data == [0.5, 0.75]


# =====================
# == LINE MANAGEMENT ==
# =====================
class TestPaintballLines:
    def test_add_lines_with_slope_tracks_named_and_unnamed_lines(self):
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])
        plot.add_lines_with_slope([0.5, 2.0], linecolor="denim", linestyle="dotted")
        plot.add_lines_with_slope([1.0], linecolor="amber", name="custom")

        assert [line.slope for line in unnamed_lines(plot)] == [0.5, 2.0]
        assert "custom" in named_lines(plot)

    def test_standard_guide_lines_are_added_by_method(self):
        # The ctor no longer seeds guide lines; the add_*_line methods every sibling uses do.
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])
        assert plot._lines == []

        plot.add_efficiency_gap_line()
        plot.add_proportionality_line()

        guides = named_lines(plot)
        assert guides["efficiency_gap"].slope == 2.0
        assert guides["efficiency_gap"].linestyle == "solid"
        assert guides["proportionality"].slope == 1.0
        assert guides["proportionality"].linestyle == "dashed"

    def test_duplicate_names_keep_both_lines(self):
        # Regression: the earlier dict storage silently dropped the first line on a name reuse.
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])
        plot.add_lines_with_slope([0.5], name="dup")
        plot.add_lines_with_slope([2.0], name="dup")

        assert [line.slope for name, line in plot._lines if name == "dup"] == [0.5, 2.0]

    def test_clear_lines_clears_named_and_unnamed_lines(self):
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])
        plot.add_lines_with_slope([0.5], name="custom")
        plot.add_lines_with_slope([1.5])
        plot.clear_lines()

        assert plot._lines == []


# =======================
# == OPTION MANAGEMENT ==
# =======================
class TestPaintballOptionsManagement:
    def test_clear_options_resets_overrides(self):
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])
        plot.set_marker_options(size=12, color="black", alpha=0.4)
        plot.set_crosshair_options(color="red", width=2.0)
        plot.clear_options()

        assert plot.options.markersize == 8.0
        assert plot.options.markercolor == "cadmiumgreen"
        assert plot.options.crosshair_color == "gray!50"
        assert plot.options.crosshair_width == 5.0

    def test_set_limits_with_rescale_updates_scales(self):
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])
        plot.set_xlim(0.25, 0.75, rescale=True)
        plot.set_ylim(0.2, 0.8, rescale=True)

        assert plot.options.xlim == (0.25, 0.75)
        assert plot.options.ylim == (0.2, 0.8)
        assert plot.options.xscale == pytest.approx(20.0)
        assert plot.options.yscale == pytest.approx(16.6667, rel=1e-4)

    def test_successive_rescales_preserve_drawn_width(self):
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])

        plot.set_xlim(0.25, 0.75, rescale=True)
        first_width = plot.options.xscale * (plot.options.xlim[1] - plot.options.xlim[0])
        plot.set_xlim(0.375, 0.625, rescale=True)
        second_width = plot.options.xscale * (plot.options.xlim[1] - plot.options.xlim[0])

        assert first_width == pytest.approx(10.0)
        assert second_width == pytest.approx(first_width)

    def test_set_scale_and_option_setters_update_all_edge_and_hull_values(self):
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])

        plot.set_scale(xscale=12.0, yscale=8.0)
        plot.set_marker_options(
            edgecolor="amber",
            edgewidth=2.0,
            edgealpha=0.6,
        )
        plot.set_hull_options(
            color="denim",
            alpha=0.3,
            edgecolor="black",
            edgewidth=1.5,
            edgealpha=0.9,
        )

        assert plot.options.xscale == 12.0
        assert plot.options.yscale == 8.0
        assert plot.options.markeredgecolor == "amber"
        assert plot.options.markeredgewidth == 2.0
        assert plot.options.markeredgealpha == 0.6
        assert plot.options.hullcolor == "denim"
        assert plot.options.hullalpha == 0.3
        assert plot.options.hulledgecolor == "black"
        assert plot.options.hulledgewidth == 1.5
        assert plot.options.hulledgealpha == 0.9

    def test_hull_options_omitted_kwargs_leave_settings_unchanged(self):
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])
        plot.set_hull_options(color="denim", edgewidth=1.5)

        plot.set_hull_options(alpha=0.3)

        assert plot.options.hullcolor == "denim"
        assert plot.options.hulledgewidth == 1.5
        assert plot.options.hullalpha == 0.3

    def test_hull_options_explicit_none_removes_fill_and_edge(self):
        plot = TikzPaintballPlot(vote_share_data=[0.4, 0.6], seats_data=[0.25, 0.75])
        plot.set_hull_options(
            color="denim", alpha=0.3, edgecolor="amber", edgewidth=1.5, edgealpha=0.9
        )

        plot.set_hull_options(
            color=None, alpha=None, edgecolor=None, edgewidth=None, edgealpha=None
        )

        assert plot.options.hullcolor is None
        assert plot.options.hullalpha is None
        assert plot.options.hulledgecolor is None
        assert plot.options.hulledgewidth is None
        assert plot.options.hulledgealpha is None

        hull = plot._paintball_hull_str()
        assert "fill=none" in hull
        assert "draw=none" in hull
        assert "fill opacity=0.8" in hull  # markeralpha
        assert "line width=0.5" in hull  # markeredgewidth
        assert "draw opacity=1.0" in hull  # markeredgealpha

    def test_marker_options_explicit_none_removes_fill_and_edge(self):
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])
        plot.set_marker_options(color=None, edgecolor=None)

        marker = plot._paintball_points_str()
        assert "fill=none" in marker
        assert "draw=none" in marker


# =======================
# == STRING GENERATION ==
# =======================
class TestPaintballStringGeneration:
    def test_inline_colors_never_mutate_the_document_color_table(self):
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])

        def inline(color):
            return plot._inline_color_value(plot._to_latex_color(color))

        assert inline("none") == "none"
        assert inline(None) == "none"
        assert inline("denim!20!amber") == "denim!20!amber"
        # Hex colors emit xcolor's extended inline specification instead of registering a
        # document-level \definecolor.
        assert inline("#123456") == "{rgb,255:red,18;green,52;blue,86}"
        assert plot.document.color_dict == {}

    def test_document_properties_update_body_string(self):
        plot = TikzPaintballPlot(vote_share_data=[0.4, 0.6], seats_data=[0.5, 0.75])

        point_document = plot.document
        hull_document = plot.hull_document

        assert point_document is not hull_document
        assert r"\begin{tikzpicture}" in point_document.body_string
        assert r"\foreach \votes/\seats" in point_document.body_string
        assert r"\begin{tikzpicture}" in hull_document.body_string
        assert r"\draw [line width=" in hull_document.body_string
        assert "fill=" in hull_document.body_string

    def test_print_emits_point_or_hull_body(self, capsys):
        plot = TikzPaintballPlot(vote_share_data=[0.4], seats_data=[0.5])

        plot.print()
        out = capsys.readouterr().out
        assert r"\begin{tikzpicture}" in out
        assert r"\foreach \votes/\seats in {" in out

        plot.print(hull=True)
        out = capsys.readouterr().out
        assert r"\draw [line width=" in out
        assert "fill=" in out

    def test_generate_latex_includes_crosshairs_lines_and_points(self):
        plot = TikzPaintballPlot(vote_share_data=[0.4], seats_data=[0.5])
        plot.add_lines_with_slope([0.5], linecolor="denim", linestyle="dotted")

        latex = str(plot)

        assert r"\begin{tikzpicture}" in latex
        assert r"\clip [draw]" in latex
        assert "line width=5.00pt" in latex
        assert "dotted" in latex
        assert r"\foreach \votes/\seats in {" in latex

    def test_crosshairs_span_configured_limits(self):
        # Regression: crosshairs used to hardcode the unit square regardless of limits.
        plot = TikzPaintballPlot(vote_share_data=[0.5], seats_data=[0.5])
        plot.set_xlim(0.25, 0.75)
        plot.set_ylim(0.2, 0.8)

        latex = plot._generate_latex()

        assert "(0.5, 0.2) -- (0.5, 0.8)" in latex
        assert "(0.25, 0.5) -- (0.75, 0.5)" in latex

    def test_hull_string_uses_hull_specific_overrides(self):
        plot = TikzPaintballPlot(vote_share_data=[0.4, 0.6], seats_data=[0.5, 0.75])
        plot.set_hull_options(
            color="denim",
            alpha=0.3,
            edgecolor="amber",
            edgewidth=1.5,
            edgealpha=0.9,
        )

        hull = plot._paintball_hull_str()
        assert "fill=denim" in hull
        assert "fill opacity=0.3" in hull
        assert "line width=1.5" in hull
        assert r"\color{amber}" in hull
        assert "draw opacity=0.9" in hull

    def test_hull_string_tracks_min_and_max_x_for_duplicate_y_values(self):
        plot = TikzPaintballPlot(
            vote_share_data=[0.5, 0.8, 0.2],
            seats_data=[0.5, 0.5, 0.5],
        )

        hull = plot._paintball_hull_str()

        assert "fill=cadmiumgreen" in hull
        assert "line width=2.0" in hull
        assert "(0.2000,0.5000)--" in hull
        assert "(0.8000,0.5000) -- cycle;" in hull


# ==================
# == EXACT OUTPUT ==
# ==================
class TestPaintballExactOutput:
    def test_points_body_is_emitted_exactly(self):
        plot = TikzPaintballPlot([0.4, 0.6], [0.25, 0.75])
        plot.add_proportionality_line()

        expected = "\n".join(
            [
                r"\begin{tikzpicture}",
                r"\begin{scope}[xscale=10.0, yscale=10.0]",
                "",
                r"\clip [draw] (0.0, 0.0) rectangle (1.0, 1.0);",
                "",
                r"{\color{gray!50}\draw [line width=5.00pt] (0.5, 0.0) -- (0.5, 1.0);}",
                r"{\color{gray!50}\draw [line width=5.00pt] (0.0, 0.5) -- (1.0, 0.5);}",
                "",
                r"{\color{gray}\draw [line width=1.00pt, dashed] "
                r"(0.0, 0.0) -- (1.0, 1.0);}",
                "",
                r"\foreach \votes/\seats in {",
                "    0.4000/0.2500,",
                "    0.6000/0.7500",
                "} {",
                r"    \node [transform shape=false, circle, inner sep=0pt, "
                r"minimum size=8.00pt, fill=cadmiumgreen, draw=cadmiumgreen, "
                r"fill opacity=0.8000, line width=0.50pt, draw opacity=1.0000] "
                r"at (\votes, \seats) {};",
                "}",
                "",
                r"\end{scope}",
                r"\end{tikzpicture}",
                "",
            ]
        )
        assert plot._generate_latex() == expected

    def test_hull_body_is_emitted_exactly(self):
        plot = TikzPaintballPlot([0.4, 0.6], [0.25, 0.75])

        expected = "\n".join(
            [
                r"\begin{tikzpicture}",
                r"\begin{scope}[xscale=10.0, yscale=10.0]",
                "",
                r"\clip [draw] (0.0, 0.0) rectangle (1.0, 1.0);",
                "",
                r"{\color{gray!50}\draw [line width=5.00pt] (0.5, 0.0) -- (0.5, 1.0);}",
                r"{\color{gray!50}\draw [line width=5.00pt] (0.0, 0.5) -- (1.0, 0.5);}",
                "",
                "",
                r"{\color{cadmiumgreen}\draw [line width=2.00pt, fill=cadmiumgreen, "
                r"fill opacity=0.8000, draw opacity=1.0000] ",
                "  (0.4000,0.2500)--",
                "  (0.6000,0.7500)--",
                "  (0.6000,0.7500)--",
                "  (0.4000,0.2500) -- cycle;}",
                "",
                r"\end{scope}",
                r"\end{tikzpicture}",
                "",
            ]
        )
        assert plot._generate_latex(hull=True) == expected


@pytest.mark.latex
def test_default_paintball_document_compiles():
    TikzPaintballPlot([0.4, 0.6], [0.25, 0.75]).document._compile_pdf()
