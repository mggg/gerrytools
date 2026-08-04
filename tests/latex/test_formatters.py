"""Tests for LaTeX cell formatter helpers."""

import math
import re

import pytest

from gerrytools.latex._colors import split_cell_fill
from gerrytools.latex.formatters import (
    _safe_round,
    boxed_center,
    compose_formatters,
    diverging_gradient_formatter,
    highlight_between,
    highlight_ge,
    highlight_gt,
    highlight_le,
    highlight_lt,
    latex_commands_for,
    round_decimals,
    wrap_with_tex_command,
)


def command_name_of(formatter) -> str:
    """Extract the generated command name from a formatter's single preamble command."""
    (command,) = latex_commands_for(formatter)
    match = re.search(r"\\newcommand\{\\([A-Za-z]+)\}", command)
    assert match is not None
    return match.group(1)


# ======================
# == BASIC FORMATTERS ==
# ======================
class TestBasicFormatters:
    def test_boxed_center_uses_width_for_default_height(self):
        formatter = boxed_center(4)
        value, rendered = formatter("raw", "X")

        assert value == "raw"
        assert rendered.startswith(r"\parbox[c][4mm][c]{4mm}")

    def test_boxed_center_wraps_content_and_preserves_value(self):
        formatter = boxed_center(3, height=5, unit="mm")
        value, rendered = formatter(7, "X")

        assert value == 7
        assert rendered.startswith(r"\parbox[c][5mm][c]{3mm}")
        assert r"\centering\strut X" in rendered

    def test_wrap_with_tex_command_wraps_rendered_string(self):
        formatter = wrap_with_tex_command("textbf")
        value, rendered = formatter("raw", "formatted")

        assert value == "raw"
        assert rendered == r"\textbf{formatted}"

    def test_compose_formatters_applies_right_to_left(self):
        formatter = compose_formatters(
            wrap_with_tex_command("textbf"),
            wrap_with_tex_command("emph"),
        )
        _value, rendered = formatter("raw", "hello")

        assert rendered == r"\textbf{\emph{hello}}"

    def test_round_decimals_formats_numeric_values_only(self):
        formatter = round_decimals(2)

        assert formatter(1.2349, "ignored") == (1.2349, "1.23")
        assert formatter("text", "text") == ("text", "text")

    def test_round_decimals_normalizes_negative_zero(self):
        formatter = round_decimals(2)

        assert formatter(-0.001, "ignored") == (-0.001, "0.00")
        assert formatter(-0.006, "ignored") == (-0.006, "-0.01")

    def test_compose_carries_fill_as_effect_past_an_outer_wrapper(self):
        # Regression: the wrapper used to receive the CellFillText as plain text, discarding the
        # fill metadata and burying the \cellcolor prefix inside \textbf.
        formatter = compose_formatters(
            wrap_with_tex_command("textbf"),
            highlight_ge(0.5, color="#00FF00", command_prefix=None),
        )

        value, rendered = formatter(0.75, "0.75")

        assert value == 0.75
        assert split_cell_fill(rendered) == ("[HTML]{00FF00}", r"\textbf{0.75}")
        assert rendered == r"\cellcolor[HTML]{00FF00}\textbf{0.75}"

    def test_compose_carries_fill_as_effect_past_an_inner_wrapper(self):
        formatter = compose_formatters(
            highlight_ge(0.5, color="#00FF00", command_prefix=None),
            wrap_with_tex_command("textbf"),
        )

        value, rendered = formatter(0.75, "0.75")

        assert value == 0.75
        assert split_cell_fill(rendered) == ("[HTML]{00FF00}", r"\textbf{0.75}")

    def test_compose_later_applied_fill_overrides_earlier(self):
        formatter = compose_formatters(
            highlight_ge(0.5, color="teal", command_prefix=None),
            highlight_ge(0.0, color="salmon", command_prefix=None),
        )

        value, rendered = formatter(0.75, "0.75")

        assert value == 0.75
        assert split_cell_fill(rendered) == ("{teal}", "0.75")

    def test_compose_formatters_preserves_latex_command_metadata(self):
        formatter = compose_formatters(
            diverging_gradient_formatter(),
            round_decimals(3),
        )

        commands = latex_commands_for(formatter)
        name = command_name_of(formatter)

        assert len(commands) == 1
        assert rf"\newcommand{{\{name}}}[1]{{%" in commands[0]

    def test_diverging_gradient_formatter_defaults_to_divgrad_stem(self):
        formatter = diverging_gradient_formatter(precision=3)

        value, rendered = formatter(0.75, "ignored")
        name = command_name_of(formatter)

        assert value == 0.75
        assert name.startswith("divgrad")
        assert rendered == rf"\{name}{{0.750}}"
        assert rf"\newcommand{{\{name}}}[1]{{%" in latex_commands_for(formatter)[0]

    def test_diverging_gradient_formatter_can_emit_compact_command_call(self):
        formatter = diverging_gradient_formatter(
            command_name="scoreheat",
            color_lo="steelblue",
            color_mid="white",
            color_hi="firebrick",
            precision=3,
        )

        value, rendered = formatter(0.75, "0.750")
        name = command_name_of(formatter)

        assert value == 0.75
        assert name.startswith("scoreheat")
        assert rendered == rf"\{name}{{0.750}}"
        assert rf"\colorlet{{{name}Losteelblue}}{{steelblue}}%" in latex_commands_for(formatter)[0]

    def test_diverging_gradient_command_uses_precision_without_rounding_formatter(self):
        formatter = diverging_gradient_formatter(command_name="scoreheat", precision=2)

        name = command_name_of(formatter)
        assert formatter(0.756, "0.756")[1] == rf"\{name}{{0.76}}"

    def test_diverging_gradient_formatter_can_emit_literal_cellcolor(self):
        formatter = diverging_gradient_formatter(command_name=None)

        assert formatter(0.75, "0.750")[1].startswith(r"\cellcolor[HTML]{")
        assert latex_commands_for(formatter) == ()

    def test_diverging_gradient_formatter_rejects_none_endpoint(self):
        with pytest.raises(ValueError, match="cannot be 'none'"):
            diverging_gradient_formatter(command_name=None, color_mid="none")

    def test_command_based_gradient_rejects_hex_endpoint(self):
        with pytest.raises(ValueError, match="cannot use hex colors"):
            diverging_gradient_formatter(color_lo="#112233")

    def test_command_based_gradient_clamps_infinities_to_endpoint_colors(self):
        # \command{inf} is a siunitx compile error, so infinities take the literal fill path
        # with the clamped endpoint color, mirroring the literal path's clamping.
        formatter = diverging_gradient_formatter(
            color_lo="steelblue", color_mid="white", color_hi="firebrick"
        )

        value, rendered = formatter(float("inf"), "inf")
        assert value == float("inf")
        assert split_cell_fill(rendered) == ("{firebrick}", "inf")

        value, rendered = formatter(float("-inf"), "-inf")
        assert value == float("-inf")
        assert split_cell_fill(rendered) == ("{steelblue}", "-inf")


# ======================
# == ROUNDING HELPERS ==
# ======================
class TestSafeRound:
    def test_safe_round_rounds_finite_numeric_values(self):
        assert _safe_round(1.2349, 2) == 1.23

    def test_safe_round_preserves_nan_and_infinities(self):
        nan_result = _safe_round(float("nan"), 2)
        assert isinstance(nan_result, float) and math.isnan(nan_result)
        assert _safe_round(float("inf"), 2) == float("inf")
        assert _safe_round(float("-inf"), 2) == float("-inf")

    def test_safe_round_preserves_non_numeric_values(self):
        assert _safe_round("hello", 2) == "hello"
        assert _safe_round(1.2349, None) == 1.2349


# ==========================
# == NUMERIC HIGHLIGHTERS ==
# ==========================
class TestNumericHighlighters:
    def test_highlight_gt_wraps_matching_values_with_command(self):
        formatter = highlight_gt(10, color="denim", command_prefix="gt")
        name = command_name_of(formatter)
        assert name.startswith("gt")
        assert formatter(11, "11") == (11, rf"\{name}{{11}}")
        assert formatter(10, "10") == (10, "10")
        assert latex_commands_for(formatter) == (
            rf"\newcommand{{\{name}}}[1]{{\cellcolor{{denim}}#1}}",
        )

    def test_highlight_ge_respects_rounding(self):
        formatter = highlight_ge(1.23, color="#00FF00", round_to=2, command_prefix="ge")
        name = command_name_of(formatter)
        assert name == "geaubhyz"
        assert formatter(1.234, "1.234") == (1.234, rf"\{name}{{1.234}}")
        assert formatter(1.225, "1.225") == (1.225, rf"\{name}{{1.225}}")
        assert latex_commands_for(formatter) == (
            rf"\newcommand{{\{name}}}[1]{{\cellcolor[HTML]{{00FF00}}#1}}",
        )

    def test_highlight_ge_can_emit_literal_cellcolor(self):
        formatter = highlight_ge(1.23, color="#00FF00", round_to=2, command_prefix=None)
        assert formatter(1.234, "1.234") == (1.234, r"\cellcolor[HTML]{00FF00}1.234")
        assert latex_commands_for(formatter) == ()

    def test_highlight_lt_and_le_leave_strings_unchanged(self):
        assert highlight_lt(5)("text", "text") == ("text", "text")
        assert highlight_le(5)("text", "text") == ("text", "text")

    def test_highlight_between_respects_inclusive_flags(self):
        inclusive = highlight_between(1, 2, color="amber", command_prefix="btw")
        exclusive = highlight_between(
            1,
            2,
            color="amber",
            include_lower=False,
            include_upper=False,
            command_prefix="btw",
        )

        name = command_name_of(inclusive)
        assert name.startswith("btw")
        assert inclusive(1, "1") == (1, rf"\{name}{{1}}")
        assert inclusive(2, "2") == (2, rf"\{name}{{2}}")
        assert exclusive(1, "1") == (1, "1")
        assert exclusive(2, "2") == (2, "2")
