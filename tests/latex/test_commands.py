"""Tests for LaTeX command helper generation."""

import pytest

from gerrytools.latex.commands import (
    _tex_ident,
    tex_cell_highlight_command,
    tex_diverging_gradient_command,
    tex_gradient_command,
    tex_twocolor_gradient_command,
    validate_command_name,
)


# =============================
# == COMMAND NAME VALIDATION ==
# =============================
class TestValidateCommandName:
    def test_validate_command_name_accepts_letters_only(self):
        validate_command_name("textbf")
        validate_command_name("Gradient")

    def test_validate_command_name_rejects_leading_backslash(self):
        with pytest.raises(ValueError, match="should not start with '\\\\\\\\'"):
            validate_command_name(r"\textbf")

    def test_validate_command_name_rejects_non_letter_tokens(self):
        for bad_name in ("cmd1", "text_bf", "", "heat-map"):
            with pytest.raises(ValueError, match="only letters"):
                validate_command_name(bad_name)


# ===========================
# == IDENTIFIER SANITIZING ==
# ===========================
class TestTexIdent:
    def test_tex_ident_strips_non_alphanumeric_characters(self):
        assert _tex_ident("red!20!blue") == "red20blue"
        assert _tex_ident("hello world") == "helloworld"

    def test_tex_ident_falls_back_to_x_for_empty_identifier(self):
        assert _tex_ident("!!!") == "X"


# ========================
# == COMMAND GENERATION ==
# ========================
class TestCommandGeneration:
    def test_gradient_command_delegates_to_twocolor_with_white_upper_endpoint(self):
        out = tex_gradient_command(cmd_str="shade", color_name="denim", lo=1.0, hi=9.0)

        assert out == tex_twocolor_gradient_command(
            "shade", lo=1.0, hi=9.0, color_lo="denim", color_hi="white"
        )
        assert r"\newcommand{\shade}[1]{%" in out
        assert r"\num[round-mode=places,round-precision=4]{#1}%" in out

    def test_twocolor_gradient_command_includes_precision_and_color_mix(self):
        out = tex_twocolor_gradient_command(
            cmd_str="heat",
            lo=-1.0,
            hi=1.0,
            color_lo="denim",
            color_hi="amber",
            precision=3,
        )

        assert r"\newcommand{\heat}[1]{%" in out
        assert r"\edef\heatlo{-1.0}\edef\heathi{1.0}%" in out
        assert r"\edef\heatpct{\fpeval{round(100*(1-\heatt),0)}}%" in out
        assert r"\edef\heatcolorspec{denim!\heatpct!amber}%" in out
        assert r"\num[round-mode=places,round-precision=3]{#1}%" in out

    def test_twocolor_gradient_command_rejects_hex_endpoint(self):
        with pytest.raises(ValueError, match="cannot use hex colors"):
            tex_twocolor_gradient_command(color_lo="#112233")

    def test_cell_highlight_command_wraps_cellcolor(self):
        out = tex_cell_highlight_command("gea", color="denim")

        assert out == r"\newcommand{\gea}[1]{\cellcolor{denim}#1}"

    def test_diverging_gradient_command_defines_colorlets_and_command(self):
        out = tex_diverging_gradient_command(
            cmd_str="heatmap",
            lo=-2.0,
            mid=0.0,
            hi=2.0,
            color_lo="darkpastelgreen",
            color_mid="white",
            color_hi="richlavender",
            precision=2,
        )

        assert r"\colorlet{heatmapLodarkpastelgreen}{darkpastelgreen}%" in out
        assert r"\colorlet{heatmapMidwhite}{white}%" in out
        assert r"\colorlet{heatmapHirichlavender}{richlavender}%" in out
        assert r"\newcommand{\heatmap}[1]{%" in out
        assert r"\num[round-mode=places,round-precision=2]{#1}%" in out

    def test_diverging_gradient_command_branches_in_fpeval_space(self):
        # Regression (C3): \ifdim on "<value> pt" overflows TeX's 16383pt dimension ceiling
        # for population-scale ranges; the side selection must stay in fpeval space.
        out = tex_diverging_gradient_command(lo=0.0, mid=50000.0, hi=100000.0)

        assert r"\ifnum\fpeval{\heatxc < \heatmid}=1" in out
        assert r"\ifdim" not in out

    def test_diverging_gradient_command_rejects_hex_endpoint(self):
        with pytest.raises(ValueError, match="cannot use hex colors"):
            tex_diverging_gradient_command(color_hi="AABBCC")

    @pytest.mark.latex
    def test_diverging_gradient_command_compiles_at_population_scale(self):
        from gerrytools.latex.document import TexDocument

        document = TexDocument()
        document.add_command(
            tex_diverging_gradient_command("popheat", lo=0.0, mid=50000.0, hi=100000.0)
        )
        # Exercise both sides of the midpoint at magnitudes far past 16383pt.
        document.body_string = (
            "\\begin{tabular}{cc}\n\\popheat{12500} & \\popheat{87500} \\\\\n\\end{tabular}"
        )
        document._compile_pdf()
