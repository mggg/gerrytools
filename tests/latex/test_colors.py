"""Tests for LaTeX color helper utilities (gerrytools/latex/_colors.py)."""

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from gerrytools.colors import LATEX_COLOR_DICT
from gerrytools.latex._colors import (
    _LATEX_COLOR_NAMES,
    CellFillText,
    cellcolor_prefix,
    classify_tikz_color,
    is_hex_color,
    is_latex_color_expression,
    normalize_hex_color,
    split_cell_fill,
    to_latex_xcolor_or_html_spec,
)
from gerrytools.latex.table import TexTable

# ===================
# == HEX UTILITIES ==
# ===================


class TestHexUtilities:
    def test_is_hex_color_accepts_hash_and_whitespace(self):
        assert is_hex_color(" #Aa00Ff ")
        assert is_hex_color("00FF00")

    def test_is_hex_color_rejects_invalid_strings(self):
        assert not is_hex_color("")
        assert not is_hex_color("#12345")
        assert not is_hex_color("#1234567")
        assert not is_hex_color("#12GG56")

    def test_normalize_hex_color_lowercases_and_strips_hash(self):
        assert normalize_hex_color(" #Aa00Ff ") == "aa00ff"

    def test_normalize_hex_color_invalid_raises(self):
        with pytest.raises(ValueError, match="HEX string"):
            normalize_hex_color("not-a-color")


# ========================
# == XCOLOR EXPRESSIONS ==
# ========================


class TestXcolorExpressions:
    def test_is_latex_color_expression_accepts_single_name(self):
        assert is_latex_color_expression("denim")

    def test_is_latex_color_expression_accepts_compound_expression(self):
        assert is_latex_color_expression("denim!25!amber!50!white")

    def test_is_latex_color_expression_rejects_empty_segments(self):
        # "red!!50" used to be silently repaired to "red!50" by the second parser.
        assert not is_latex_color_expression("red!!50")

    def test_is_latex_color_expression_rejects_unknown_names(self):
        assert not is_latex_color_expression("notacolor!25!amber")

    def test_is_latex_color_expression_rejects_out_of_range_percentages(self):
        assert not is_latex_color_expression("denim!101!amber")
        assert not is_latex_color_expression("denim!-1!amber")

    def test_is_latex_color_expression_rejects_non_numeric_percentage(self):
        # "abc" cannot be parsed as float → returns False
        assert not is_latex_color_expression("denim!abc!amber")

    def test_is_latex_color_expression_rejects_empty_or_malformed_tokens(self):
        assert not is_latex_color_expression("")
        assert not is_latex_color_expression("!!!")
        assert not is_latex_color_expression("denim!!amber")

    def test_latex_color_names_match_exact_squashed_keys(self):
        assert is_latex_color_expression("cadmiumgreen")
        assert not is_latex_color_expression("Cadmium Green")
        assert not is_latex_color_expression("Cadmiumgreen")


# ========================
# == COLOR SPEC PARSING ==
# ========================


class TestColorSpecParsing:
    def test_tikz_classifier_rejects_nonopaque_alpha(self):
        with pytest.raises(ValueError, match="non-opaque alpha"):
            classify_tikz_color("#ff000080")

    def test_overridden_latex_name_uses_resolved_html_color(self):
        assert "classicrose" not in _LATEX_COLOR_NAMES
        assert classify_tikz_color("classicrose") == ("html", "FCCDE5")

    def test_to_latex_xcolor_or_html_spec_handles_255_scale_rgb_tuple(self):
        assert to_latex_xcolor_or_html_spec((12, 34, 56)) == ("RGB", (12, 34, 56))
        assert to_latex_xcolor_or_html_spec((255.0, 127.6, 0.4)) == ("RGB", (255, 128, 0))

    def test_to_latex_xcolor_or_html_spec_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="tuple of length 3"):
            to_latex_xcolor_or_html_spec((1, 2))  # type: ignore

    def test_to_latex_xcolor_or_html_spec_invalid_range_raises(self):
        with pytest.raises(ValueError, match="range \\[0\\.0, 1\\.0\\] or \\[0, 255\\]"):
            to_latex_xcolor_or_html_spec((300, 0, 0))

    def test_to_latex_xcolor_or_html_spec_rejects_boolean_components(self):
        # Regression: (True, 0, 0) was accepted as unit-scale red.
        with pytest.raises(ValueError, match="boolean"):
            to_latex_xcolor_or_html_spec((True, 0, 0))

    def test_to_latex_xcolor_or_html_spec_rejects_ambiguous_scale(self):
        # Regression: (1.5, 0.5, 0.5) was silently mangled to the 255-scale tuple (2, 0, 0)
        # instead of hitting the canonical model's ambiguity guard.
        with pytest.raises(ValueError, match="Ambiguous RGB tuple"):
            to_latex_xcolor_or_html_spec((1.5, 0.5, 0.5))

    def test_to_latex_xcolor_or_html_spec_preserves_xcolor_expressions(self):
        assert to_latex_xcolor_or_html_spec("denim") == ("NAME", "denim")
        assert to_latex_xcolor_or_html_spec("denim!25!amber") == ("NAME", "denim!25!amber")

    def test_to_latex_xcolor_or_html_spec_converts_hex_string_to_html(self):
        # A bare hex string bypasses the xcolor-expression check and goes through is_hex_color
        assert to_latex_xcolor_or_html_spec("#AABBCC") == ("HTML", "aabbcc")

    def test_to_latex_xcolor_or_html_spec_converts_named_colors_to_html(self):
        assert to_latex_xcolor_or_html_spec("tab:blue") == ("HTML", "1f77b4")

    @pytest.mark.parametrize("color", ["Cadmium Green", "Cadmiumgreen"])
    def test_to_latex_xcolor_or_html_spec_routes_nonexact_names_to_html(self, color):
        color_type, _ = to_latex_xcolor_or_html_spec(color)

        assert color_type == "HTML"

    def test_to_latex_xcolor_or_html_spec_preserves_exact_squashed_name(self):
        assert to_latex_xcolor_or_html_spec("cadmiumgreen") == ("NAME", "cadmiumgreen")

    @pytest.mark.parametrize("name", ["tenne(tawny)", "olivedrab7", "olivedrab(web)(olivedrab3)"])
    def test_package_defined_squashed_names_take_the_name_path(self, name):
        # Regression: these package-defined names were unreachable (the dict stored
        # "tenné(tawny)" and "#"-bearing keys).
        assert to_latex_xcolor_or_html_spec(name) == ("NAME", name)

    @pytest.mark.parametrize(
        "name",
        [
            "davy'sgrey",
            "payne'sgrey",
            "hooker'sgreen",
            "st.patrick'sblue",
            "screamin'green",
            "tiger'seye",
        ],
    )
    def test_apostrophe_names_route_to_html_fallback(self, name):
        # The package defines these with a literal backslash-apostrophe (payne\'sgrey), so
        # emitting the plain key as ("NAME", ...) would be an undefined color at compile.
        color_type, color_value = to_latex_xcolor_or_html_spec(name)

        assert color_type == "HTML"
        assert color_value == LATEX_COLOR_DICT[name].lstrip("#")

    def test_to_latex_xcolor_or_html_spec_rejects_non_opaque_alpha(self):
        with pytest.raises(ValueError, match="non-opaque alpha"):
            to_latex_xcolor_or_html_spec("#11223344")

    def test_to_latex_xcolor_or_html_spec_truncates_fully_opaque_alpha(self):
        assert to_latex_xcolor_or_html_spec("#112233FF") == ("HTML", "112233")

    def test_to_latex_xcolor_or_html_spec_rejects_none(self):
        with pytest.raises(ValueError, match="cannot be emitted"):
            to_latex_xcolor_or_html_spec("none")

    def test_to_latex_xcolor_or_html_spec_handles_unit_rgb_tuple(self):
        assert to_latex_xcolor_or_html_spec((0.1, 0.2, 0.3)) == ("rgb", (0.1, 0.2, 0.3))


# ======================
# == CELLCOLOR PREFIX ==
# ======================


class TestCellcolorPrefix:
    def test_cellcolor_prefix_supports_name_html_rgb_and_RGB(self):
        assert cellcolor_prefix("denim!20!amber") == r"\cellcolor{denim!20!amber}"
        assert cellcolor_prefix("#ABCDEF") == r"\cellcolor[HTML]{ABCDEF}"
        assert cellcolor_prefix((0.1, 0.2, 0.3)) == r"\cellcolor[rgb]{0.100,0.200,0.300}"
        assert cellcolor_prefix((10, 20, 30)) == r"\cellcolor[RGB]{10,20,30}"

    def test_cellcolor_prefix_rejects_none(self):
        with pytest.raises(ValueError, match="cannot be emitted"):
            cellcolor_prefix("none")


@pytest.mark.latex
@pytest.mark.parametrize(
    "color",
    [
        "Cadmium Green",
        "Cadmiumgreen",
        "cadmiumgreen",
        # Package-defined names that were previously unreachable through the NAME path.
        "tenne(tawny)",
        "olivedrab7",
        "olivedrab(web)(olivedrab3)",
        # Apostrophe names must compile via the HTML fallback; emitted by name they are
        # undefined colors (the package spells them with a backslash-apostrophe).
        "davy'sgrey",
        "payne'sgrey",
        "screamin'green",
    ],
)
def test_latex_color_name_classifications_compile(color):
    table = TexTable(pd.DataFrame({"value": [1]}), use_defaults=False)
    table.highlight_rows(0, color=color)

    table.document._compile_pdf()


# ==============================
# == LATEXCOLORS.STY PARITY ==
# ==============================


def _installed_latexcolors_sty() -> Path | None:
    """Locate the installed latexcolors.sty via kpsewhich, or None when unavailable."""
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        return None
    result = subprocess.run(
        [kpsewhich, "latexcolors.sty"], capture_output=True, text=True, timeout=30
    )
    sty_path_text = result.stdout.strip()
    if result.returncode != 0 or not sty_path_text:
        return None
    return Path(sty_path_text)


@pytest.mark.latex
def test_latex_color_dict_matches_installed_latexcolors_sty():
    """Diff LATEX_COLOR_DICT against the installed package so drift is caught.

    Every ``\\definecolor{name}{rgb}{...}`` in the sty must have a dict entry (mapping the
    package's ``\\'`` spelling to the plain-apostrophe key) whose hex value reproduces the sty's
    2-decimal rgb components, and every NAME-path-emittable key must be package-defined.
    """
    sty_path = _installed_latexcolors_sty()
    if sty_path is None:
        pytest.skip("kpsewhich cannot locate latexcolors.sty")

    definitions = re.findall(r"\\definecolor\{([^}]*)\}\{rgb\}\{([^}]*)\}", sty_path.read_text())
    assert len(definitions) > 700  # the package defines ~758 colors

    package_names = {name for name, _ in definitions}
    emittable_but_undefined = _LATEX_COLOR_NAMES - package_names
    assert emittable_but_undefined == set(), (
        f"NAME-path-emittable keys the package does not define: {sorted(emittable_but_undefined)}"
    )

    for package_name, rgb_text in definitions:
        squashed_key = package_name.replace(r"\'", "'")
        assert squashed_key in LATEX_COLOR_DICT, f"missing dict key for {package_name!r}"
        hex_digits = LATEX_COLOR_DICT[squashed_key].lstrip("#")
        hex_components = [int(hex_digits[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        sty_components = [float(component) for component in rgb_text.split(",")]
        for hex_component, sty_component in zip(hex_components, sty_components, strict=True):
            assert abs(hex_component - sty_component) < 0.005 + 1e-9, (
                f"{package_name!r}: dict hex #{hex_digits} diverges from sty rgb {rgb_text}"
            )


# ====================
# == CELL FILL TEXT ==
# ====================


class TestCellFillText:
    def test_cell_fill_text_is_the_full_prefixed_string(self):
        text = CellFillText("[HTML]{ABCDEF}", "0.75")

        assert text == r"\cellcolor[HTML]{ABCDEF}0.75"
        assert text.fill_spec == "[HTML]{ABCDEF}"
        assert text.fill_text == "0.75"

    def test_split_cell_fill_recovers_parts_without_parsing(self):
        assert split_cell_fill(CellFillText("{teal}", "x")) == ("{teal}", "x")
        assert split_cell_fill(r"\cellcolor{teal}x") == (None, r"\cellcolor{teal}x")
        assert split_cell_fill("plain") == (None, "plain")
