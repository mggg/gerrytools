"""Tests for non-visual LaTeX seats-votes generation."""

from __future__ import annotations

import numpy as np
import pytest

from gerrytools.latex._tikz_plot_base import _TikzColorToken, _to_tikz_linestyle
from gerrytools.latex.seatsvotes import (
    SeatsVotesOptions,
    TikzSeatsVotesPlot,
    _SeatsVotesData,
    _SVPlotLine,
)


# ===============
# == UTILITIES ==
# ===============
class TestSeatsVotesUtilities:
    def test_to_tikz_linestyle_maps_matplotlib_tokens(self):
        assert _to_tikz_linestyle("-") == "solid"
        assert _to_tikz_linestyle("--") == "dashed"
        assert _to_tikz_linestyle(":") == "dotted"
        assert _to_tikz_linestyle("-.") == "dashdotted"
        assert _to_tikz_linestyle("dashdot") == "dashdotted"

    def test_to_tikz_linestyle_preserves_valid_tikz_styles(self):
        assert _to_tikz_linestyle("loosely dashed") == "loosely dashed"
        assert _to_tikz_linestyle("densely dotted") == "densely dotted"

    def test_to_tikz_linestyle_rejects_unknown_tokens(self):
        with pytest.raises(ValueError, match="Invalid linestyle"):
            _to_tikz_linestyle("wavy")


# ============================
# == DATA CLASSES & OPTIONS ==
# ============================
class TestSeatsVotesData:
    def test_curve_values_compute_expected_breakpoints(self):
        data = _SeatsVotesData(
            pov_party_vote_counts=np.array([40.0, 60.0]),
            total_vote_counts=np.array([100.0, 100.0]),
            name="Election",
            linecolor="grey",
            markercolor="gold",
            marker_label="Result",
        )

        vote_shares, seat_shares = data.seats_votes_curve_values()

        assert vote_shares == pytest.approx([0.0, 0.4, 0.6, 1.0])
        assert seat_shares == pytest.approx([0.0, 0.0, 0.5, 1.0])

    def test_curve_values_reject_shape_mismatch(self):
        data = _SeatsVotesData(
            pov_party_vote_counts=np.array([1.0, 2.0]),
            total_vote_counts=np.array([3.0]),
            name="Election",
            linecolor="grey",
            markercolor="gold",
            marker_label="Result",
        )

        with pytest.raises(ValueError, match="must have the same shape"):
            data.seats_votes_curve_values()

    def test_curve_values_reject_nonpositive_totals(self):
        data = _SeatsVotesData(
            pov_party_vote_counts=np.array([1.0, 2.0]),
            total_vote_counts=np.array([3.0, 0.0]),
            name="Election",
            linecolor="grey",
            markercolor="gold",
            marker_label="Result",
        )

        with pytest.raises(ValueError, match="must be positive"):
            data.seats_votes_curve_values()


class TestSeatsVotesLineAndOptions:
    def test_sv_plot_line_accepts_finite_numeric_values(self):
        line = _SVPlotLine(slope=2, linecolor="grey", linewidth=1, linestyle="--")

        assert line.slope == 2.0
        assert line.linewidth == 1.0

    def test_sv_plot_line_rejects_nan_slope(self):
        with pytest.raises(ValueError, match="slope must not be NaN"):
            _SVPlotLine(
                slope=float("nan"),
                linecolor="grey",
                linewidth=1.0,
                linestyle="--",
            )

    def test_sv_plot_line_rejects_invalid_linewidth(self):
        with pytest.raises(ValueError, match="linewidth must be finite"):
            _SVPlotLine(
                slope=1.0,
                linecolor="grey",
                linewidth=float("inf"),
                linestyle="--",
            )

        with pytest.raises(ValueError, match="linewidth must be nonnegative"):
            _SVPlotLine(
                slope=1.0,
                linecolor="grey",
                linewidth=-1.0,
                linestyle="--",
            )

    def test_options_validate_crosshair_and_axis_ranges(self):
        options = SeatsVotesOptions()

        with pytest.raises(ValueError, match="crosshair_x_width must be finite"):
            options.crosshair_x_width = float("nan")

        with pytest.raises(ValueError, match="crosshair_y_width must be nonnegative"):
            options.crosshair_y_width = -0.1

        with pytest.raises(ValueError, match=r"crosshair_alpha must be in \[0, 1\]"):
            options.crosshair_alpha = 1.5

        with pytest.raises(ValueError, match=r"xlim\[0\] must be less than xlim\[1\]"):
            options.xlim = (1.0, 0.0)

        with pytest.raises(ValueError, match="xscale must be positive"):
            options.xscale = 0.0

        with pytest.raises(ValueError, match="yscale must be finite"):
            options.yscale = float("inf")

        with pytest.raises(ValueError, match="markersize must be nonnegative"):
            options.markersize = -1.0

        with pytest.raises(ValueError, match="legend_fontsize must be finite"):
            options.legend_fontsize = float("nan")

    def test_options_reject_unknown_attributes(self):
        options = SeatsVotesOptions()

        with pytest.raises(AttributeError, match="Unknown SeatsVotesOptions attribute"):
            options.unknown = 1


# ===========================
# == CONSTRUCTION & CONFIG ==
# ===========================
class TestSeatsVotesConstruction:
    def test_legend_is_disabled_by_default(self):
        assert TikzSeatsVotesPlot().legend is False

    def test_initialization_sets_scale_and_tikz_package(self):
        plot = TikzSeatsVotesPlot(legend=True)
        plot.set_scale(xscale=12, yscale=8)

        assert plot.options.xscale == 12.0
        assert plot.options.yscale == 8.0
        assert "tikz" in plot.document.package_list
        assert plot.legend is True

    def test_add_seat_votes_data_uses_default_names_and_colors(self):
        plot = TikzSeatsVotesPlot()
        plot.add_election([0.4, 0.6])

        series = plot._sv_data_list[0]
        assert series.name == "Election Seats-Votes Curve"
        assert series.linecolor == plot.standard_election_color
        assert series.markercolor == plot.standard_marker_color
        assert series.marker_label == "Election Result"

    def test_add_election_explicit_none_removes_curve_and_marker_colors(self):
        plot = TikzSeatsVotesPlot()
        plot.add_election([0.4, 0.6], linecolor=None, markercolor=None)

        generated = str(plot)
        assert generated.count("draw=none") >= 2

    def test_none_standard_colors_remove_default_curve_and_marker(self):
        plot = TikzSeatsVotesPlot()
        plot.standard_election_color = None
        plot.standard_marker_color = None
        plot.add_election([0.4, 0.6])

        generated = str(plot)
        assert generated.count("draw=none") >= 2

    def test_add_seat_votes_data_requires_unit_interval_shares_without_totals(self):
        plot = TikzSeatsVotesPlot()

        with pytest.raises(ValueError, match="must be vote shares in \\[0, 1\\]"):
            plot.add_election([0.4, 1.2])

    def test_add_election_rejects_shape_mismatch(self):
        # Series are validated at add time, not first at render.
        plot = TikzSeatsVotesPlot()

        with pytest.raises(ValueError, match="must have the same shape"):
            plot.add_election([40, 60], [100])

    def test_add_election_rejects_nonpositive_total_votes(self):
        plot = TikzSeatsVotesPlot()

        with pytest.raises(ValueError, match="must be positive"):
            plot.add_election([40, 60], [100, 0])

    def test_add_efficiency_gap_line_and_basic_setters_update_plot_state(self):
        plot = TikzSeatsVotesPlot()

        plot.add_efficiency_gap_line()
        plot.set_label_fontsize(13.0)
        plot.set_markersize(9.5)
        plot.set_linewidth(2.25)

        assert plot._line_data_list[-1].label == "Efficiency Gap"
        assert plot.options.fontsize == 13.0
        assert plot.options.legend_fontsize == 16.0
        assert plot.options.markersize == 9.5
        assert plot.options.linewidth == 2.25

    def test_set_limits_can_rescale_axes(self):
        plot = TikzSeatsVotesPlot()

        plot.set_xlim(0.25, 0.75, rescale=True)
        plot.set_ylim(0.1, 0.9, rescale=True)

        assert plot.options.xlim == (0.25, 0.75)
        assert plot.options.ylim == (0.1, 0.9)
        assert plot.options.xscale == pytest.approx(20.0)
        assert plot.options.yscale == pytest.approx(12.5)

    def test_clear_options_resets_defaults_and_crosshairs(self):
        plot = TikzSeatsVotesPlot()
        plot.set_scale(xscale=12, yscale=8)
        plot.set_fontsize(20)
        plot.set_xlim(0.2, 0.8)
        plot.set_ylim(0.3, 0.7)
        plot.remove_crosshairs()
        plot.display_election_markers(False)
        plot.display_additional_lines_in_legend(False)

        plot.clear_options()

        assert plot.options.fontsize == 16.0
        assert plot.options.legend_fontsize == 16.0
        assert plot.options.xscale == 10.0
        assert plot.options.yscale == 10.0
        assert plot._show_crosshairs is True
        assert plot._display_election_markers is True
        assert plot._display_line_legend is True

    def test_crosshair_settings_and_visibility_flags_can_be_toggled(self):
        plot = TikzSeatsVotesPlot(legend=True)
        plot.update_crosshair_settings(
            x_width=0.1,
            y_width=0.2,
            color="denim",
            alpha=0.4,
        )

        assert plot.options.crosshair_x_width == 0.1
        assert plot.options.crosshair_y_width == 0.2
        assert plot.options.crosshair_color == "denim"
        assert plot.options.crosshair_alpha == 0.4
        assert plot._show_crosshairs is True

        plot.remove_crosshairs()
        plot.display_election_markers(False)
        plot.display_additional_lines_in_legend(False)
        assert plot._show_crosshairs is False
        assert plot._display_election_markers is False
        assert plot._display_line_legend is False

        plot.display_election_markers(True)
        plot.display_additional_lines_in_legend(True)
        assert plot._display_election_markers is True
        assert plot._display_line_legend is True


# =======================
# == INTERNAL BUILDERS ==
# =======================
class TestSeatsVotesInternalBuilders:
    def test_fontsize_command_uses_baseline_skip_plus_two_points(self):
        assert TikzSeatsVotesPlot._fontsize_command(16.0) == r"\fontsize{16.00}{18.00}\selectfont "

    def test_step_path_builds_expected_tikz_segments(self):
        path = TikzSeatsVotesPlot._step_path([0.0, 0.4, 0.6], [0.0, 0.5, 1.0])

        assert path == (
            "(0.0000, 0.0000) -- (0.0000, 0.5000) -- (0.4000, 0.5000) -- "
            "(0.4000, 1.0000) -- (0.6000, 1.0000)"
        )

    def test_step_path_rejects_empty_or_mismatched_vectors(self):
        with pytest.raises(ValueError, match="must have same length"):
            TikzSeatsVotesPlot._step_path([0.0], [0.0, 1.0])

        with pytest.raises(ValueError, match="must not be empty"):
            TikzSeatsVotesPlot._step_path([], [])

    def test_legend_entry_helpers_deduplicate_curve_and_marker_entries(self):
        plot = TikzSeatsVotesPlot()
        plot.add_election(
            [0.4, 0.6],
            name="Election A",
            linecolor="denim",
            markercolor="amber",
            marker_label="Result A",
        )
        plot.add_election(
            [0.5, 0.7],
            name="Election A",
            linecolor="denim",
            markercolor="amber",
            marker_label="Result A",
        )

        assert plot._curve_legend_entries() == [("denim", "Election A")]
        assert plot._marker_legend_entries() == [("amber", "Result A")]

    def test_line_legend_entries_skip_unlabeled_lines(self):
        plot = TikzSeatsVotesPlot()
        plot.add_proportionality_line(name="Prop")
        plot.add_custom_line(
            slope=0.5,
            linecolor="grey",
            linestyle="--",
            linewidth=1.0,
        )

        assert plot._line_legend_entries() == [("gray", "dashed", "Prop")]

    def test_color_helpers_support_xcolor_html_and_none(self):
        plot = TikzSeatsVotesPlot()

        assert plot._to_latex_color("denim!20!amber") == _TikzColorToken(
            kind="xcolor",
            value="denim!20!amber",
        )
        assert plot._to_latex_color("#ab12cd") == _TikzColorToken(
            kind="html",
            value="AB12CD",
        )
        assert plot._to_latex_color("none") == _TikzColorToken(
            kind="none",
            value="none",
        )

        assert (
            TikzSeatsVotesPlot._color_prefix(_TikzColorToken(kind="html", value="AB12CD"))
            == r"\color[HTML]{AB12CD}"
        )
        assert (
            TikzSeatsVotesPlot._color_prefix(_TikzColorToken(kind="xcolor", value="denim"))
            == r"\color{denim}"
        )
        assert TikzSeatsVotesPlot._color_prefix(_TikzColorToken(kind="none", value="none")) == ""

        assert plot._to_latex_color(None) == _TikzColorToken(  # type: ignore[arg-type]
            kind="none",
            value="none",
        )

    def test_legend_constants_scale_with_axis_span(self):
        # Regression (C6): the legend used raw data-unit constants, so non-unit limits pushed
        # it out of proportion. With xlim (0, 2): x_start = 2 + 0.03*2 = 2.06, line length
        # 0.06*2 = 0.12, label offset 0.08*2 = 0.16.
        plot = TikzSeatsVotesPlot(legend=True)
        plot.add_election([0.4, 0.6], name="Race")
        plot.set_xlim(0.0, 2.0)

        legend_lines: list[str] = []
        plot._add_legend(legend_lines)
        legend = "\n".join(legend_lines)

        assert "(2.0600, 0.5300) -- (2.1800, 0.5300)" in legend
        assert "(2.2200, 0.5300)" in legend

    def test_wrap_with_color_scope_only_wraps_colored_commands(self):
        command = r"\draw [line width=1.00pt] (0,0) -- (1,1);"

        assert (
            TikzSeatsVotesPlot._wrap_with_color_scope(
                command,
                _TikzColorToken(kind="xcolor", value="denim"),
            )
            == r"{\color{denim}\draw [line width=1.00pt] (0,0) -- (1,1);}"
        )
        assert (
            TikzSeatsVotesPlot._wrap_with_color_scope(
                command,
                _TikzColorToken(kind="none", value="none"),
            )
            == command
        )

    def test_draw_fill_and_marker_commands_encode_none_and_html_colors(self):
        plot = TikzSeatsVotesPlot()

        assert plot._draw_path_command(
            path="(0.0, 0.0) -- (1.0, 1.0)",
            color=_TikzColorToken(kind="html", value="AB12CD"),
            linewidth=1.5,
            linestyle="dashed",
        ) == (
            r"{\color[HTML]{AB12CD}\draw [line width=1.50pt, dashed] " r"(0.0, 0.0) -- (1.0, 1.0);}"
        )

        assert plot._fill_rectangle_command(
            xmin=0.0,
            ymin=0.1,
            xmax=0.2,
            ymax=0.3,
            color=_TikzColorToken(kind="none", value="none"),
            fill_opacity=0.4,
        ) == (
            r"\fill [fill opacity=0.4000, fill=none] (0.0000, 0.1000) rectangle "
            r"(0.2000, 0.3000);"
        )

        assert plot._marker_node_command(
            x=0.2,
            y=0.3,
            color=_TikzColorToken(kind="none", value="none"),
            size_pt=8.0,
        ) == (
            r"\node [circle, inner sep=0pt, minimum size=8.00pt, fill=none, draw=none] "
            r"at (0.2000, 0.3000) {};"
        )

        assert (
            plot._draw_path_command(
                path="(0.0, 0.0) -- (1.0, 1.0)",
                color=_TikzColorToken(kind="none", value="none"),
                linewidth=2.0,
            )
            == r"\draw [line width=2.00pt, draw=none] (0.0, 0.0) -- (1.0, 1.0);"
        )


# ======================
# == LATEX GENERATION ==
# ======================
class TestSeatsVotesLatexGeneration:
    def test_latex_seatsvotes_generates_tikz_with_escaped_labels_and_legend_entries(
        self,
    ):
        plot = TikzSeatsVotesPlot(
            legend=True,
            xlabel="Vote & Share",
            ylabel="Seat_Share",
            title="SV 100%",
        )
        plot.add_election(
            target_party_vote_shares=[100, 200, 300, 400],
            total_votes=[220, 390, 540, 700],
            name="Election_A",
            linecolor="#1560bd",
            markercolor="denim!20!amber",
            marker_label="Result#1",
        )
        plot.add_election(
            target_party_vote_shares=[0.40, 0.52, 0.63, 0.57],
            name="Election_A",
            linecolor="#1560bd",
            markercolor="denim!20!amber",
            marker_label="Result#1",
        )
        plot.add_proportionality_line()
        plot.add_custom_line(
            slope=0.5,
            linecolor="amber!20!denim",
            linestyle="-.",
            linewidth=1.5,
            name="Guide_1",
        )

        latex = plot.document.body_string

        assert r"\begin{tikzpicture}" in latex
        assert r"\documentclass" not in latex
        assert r"\color[HTML]{1560BD}" in latex
        assert "HTML:" not in latex
        assert r"\color{amber!20!denim}" in latex
        assert r"\color{denim!20!amber}" in latex
        assert "Election\\_A" in latex
        assert latex.count("Election\\_A") == 1
        assert "Result\\#1" in latex
        assert latex.count("Result\\#1") == 1
        assert "Guide\\_1" in latex
        assert "Proportionality" in latex
        assert "Vote \\& Share" in latex
        assert "Seat\\_Share" in latex
        assert "SV 100\\%" in latex

        full_document = str(plot.document)
        assert r"\documentclass[border=2pt]{standalone}" in full_document
        assert plot.document.body_string == latex

    def test_latex_seatsvotes_can_hide_crosshairs_markers_and_line_legend(self):
        plot = TikzSeatsVotesPlot(legend=True)
        plot.add_election(
            target_party_vote_shares=[0.48, 0.52, 0.61, 0.44],
            name="Shares",
        )
        plot.add_custom_line(
            slope=1.5,
            linecolor="denim",
            linestyle="dashdot",
            linewidth=1.0,
            name="Guide",
        )
        plot.remove_crosshairs()
        plot.display_election_markers(False)
        plot.display_additional_lines_in_legend(False)

        latex = str(plot)

        assert "fill opacity=" not in latex
        assert "minimum size=" not in latex
        assert "Guide" not in latex
        assert "dashdotted" in latex

    def test_print_emits_raw_tikz_body(self, capsys):
        plot = TikzSeatsVotesPlot()
        plot.add_election([0.4, 0.6])

        plot.print()

        out = capsys.readouterr().out
        assert r"\begin{tikzpicture}" in out
        assert r"\draw [line width=" in out
        assert r"\end{tikzpicture}" in out


# ==================
# == EXACT OUTPUT ==
# ==================
class TestSeatsVotesExactOutput:
    def test_body_is_emitted_exactly(self):
        plot = TikzSeatsVotesPlot(legend=True, title="Title & Co", xlabel="Vote share")
        plot.add_election([0.31, 0.62], name="Race")
        plot.add_proportionality_line()

        expected = "\n".join(
            [
                r"\begin{tikzpicture}",
                r"\begin{scope}[xscale=10.0, yscale=10.0]",
                r"\clip [draw] (0.0000, 0.0000) rectangle (1.0000, 1.0000);",
                "",
                r"{\color[HTML]{D3D3D3}\fill [fill opacity=1.0000] "
                r"(0.4900, 0.0000) rectangle (0.5100, 1.0000);}",
                r"{\color[HTML]{D3D3D3}\fill [fill opacity=1.0000] "
                r"(0.0000, 0.4900) rectangle (1.0000, 0.5100);}",
                "",
                r"{\color{gray}\draw [line width=1.00pt, dashed] "
                r"(0.0000, 0.0000) -- (1.0000, 1.0000);}",
                "",
                r"{\color[HTML]{006400}\draw [line width=1.50pt] (0.0000, 0.0000) -- "
                r"(0.0000, 0.0000) -- (0.3450, 0.0000) -- (0.3450, 0.5000) -- "
                r"(0.6550, 0.5000) -- (0.6550, 1.0000) -- (1.0000, 1.0000);}",
                "",
                r"{\color[HTML]{DAA520}\node [circle, inner sep=0pt, minimum size=8.00pt, "
                r"fill, draw] at (0.4650, 0.5000) {};}",
                r"\end{scope}",
                r"\begin{scope}[xscale=10.0, yscale=10.0]",
                r"\node [anchor=south] at (0.5000, 1.0300) "
                r"{\fontsize{16.00}{18.00}\selectfont Title \& Co};",
                r"\node [anchor=north] at (0.5000, -0.0300) "
                r"{\fontsize{16.00}{18.00}\selectfont Vote share};",
                r"\end{scope}",
                r"\begin{scope}[xscale=10.0, yscale=10.0]",
                r"{\color[HTML]{006400}\draw [line width=1.20pt, solid] "
                r"(1.0300, 0.5600) -- (1.0900, 0.5600);}",
                r"\node [anchor=west] at (1.1100, 0.5600) "
                r"{\fontsize{16.00}{18.00}\selectfont Race};",
                r"{\color[HTML]{DAA520}\node [circle, inner sep=0pt, minimum size=8.00pt, "
                r"fill, draw] at (1.0600, 0.5000) {};}",
                r"\node [anchor=west] at (1.1100, 0.5000) "
                r"{\fontsize{16.00}{18.00}\selectfont Election Result};",
                r"{\color{gray}\draw [line width=1.20pt, dashed] "
                r"(1.0300, 0.4400) -- (1.0900, 0.4400);}",
                r"\node [anchor=west] at (1.1100, 0.4400) "
                r"{\fontsize{16.00}{18.00}\selectfont Proportionality};",
                r"\end{scope}",
                r"\end{tikzpicture}",
                "",
            ]
        )
        assert plot._generate_latex() == expected
