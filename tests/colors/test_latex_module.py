import pytest

from gerrytools.colors.latex import (
    _rgb_to_hex,
    _xcolor_mix_hex,
    get_color_from_latex_string,
    hex_to_rgb,
    normalize_hex_color,
)

# =========================
# == HEX NORMALIZATION ==
# =========================


class TestNormalizeHex:
    def test_normalize_hex_color_expands_three_digit_values(self):
        assert normalize_hex_color("#AbC") == "#aabbcc"

    def test_normalize_hex_color_lowercases_six_digit_values(self):
        assert normalize_hex_color("A1B2C3") == "#a1b2c3"

    def test_normalize_hex_color_warns_and_strips_alpha_for_four_digit_values(self):
        with pytest.warns(UserWarning, match="Ignoring alpha channel"):
            assert normalize_hex_color("#abcd") == "#aabbcc"

    def test_normalize_hex_color_warns_and_strips_alpha_for_eight_digit_values(self):
        with pytest.warns(UserWarning, match="Ignoring alpha channel"):
            assert normalize_hex_color("#11223344") == "#112233"

    def test_normalize_hex_color_invalid_value_raises(self):
        with pytest.raises(ValueError, match="Not a valid hex color"):
            normalize_hex_color("xyz")


# ======================
# == HEX/RGB HELPERS ==
# ======================


class TestHexRgbHelpers:
    def test_hex_to_rgb_returns_rgb_tuple(self):
        assert hex_to_rgb("#123456") == (18, 52, 86)

    def test_rgb_to_hex_rounds_and_clamps_components(self):
        assert _rgb_to_hex((255.0, 127.6, -5.0)) == "#ff8000"

    def test_xcolor_mix_hex_applies_xcolor_weighting(self):
        assert _xcolor_mix_hex(["#ff0000", "#ffffff"], [50]) == "#ff8080"

    def test_xcolor_mix_hex_requires_one_more_color_than_percentages(self):
        with pytest.raises(ValueError, match="one more than number of percentages"):
            _xcolor_mix_hex(["#ff0000"], [50])


# ========================
# == LATEX RESOLUTION ==
# ========================


class TestTokenizeXcolorExpression:
    def test_names_and_percents_split_by_position(self):
        from gerrytools.colors import tokenize_xcolor_expression

        names, percents = tokenize_xcolor_expression("denim!25!amber!50!white")
        assert names == ["denim", "amber", "white"]
        assert percents == [25.0, 50.0]

    def test_trailing_percent_is_reported_by_parity(self):
        from gerrytools.colors import tokenize_xcolor_expression

        names, percents = tokenize_xcolor_expression("red!50")
        assert len(names) == len(percents) == 1

    def test_empty_segment_is_rejected_not_repaired(self):
        # "red!!50" used to be silently repaired to "red!50".
        from gerrytools.colors import tokenize_xcolor_expression

        with pytest.raises(ValueError, match="empty segment"):
            tokenize_xcolor_expression("red!!50")

    def test_non_numeric_percent_rejected(self):
        from gerrytools.colors import tokenize_xcolor_expression

        with pytest.raises(ValueError, match="not a percentage"):
            tokenize_xcolor_expression("red!blue!green!yellow")

    def test_out_of_range_percent_rejected(self):
        from gerrytools.colors import tokenize_xcolor_expression

        with pytest.raises(ValueError, match=r"\[0,100\]"):
            tokenize_xcolor_expression("red!150!blue")


class TestGetColorFromLatexString:
    def test_single_name_uses_bright_green_override(self):
        assert get_color_from_latex_string("green") == "#00ff00"

    def test_name_with_percent_defaults_to_white(self):
        assert get_color_from_latex_string("red!50") == "#ff8080"

    def test_explicit_two_color_mix(self):
        assert get_color_from_latex_string("red!0!blue") == "#0000ff"

    def test_three_color_left_fold_mix(self):
        # "red!100!blue!0!white" == left-fold: (red!100!blue)!0!white
        # red!100!blue = pure red; then red!0!white = white
        assert get_color_from_latex_string("red!100!blue!0!white") == "#ffffff"

    def test_unknown_color_name_raises(self):
        with pytest.raises(KeyError, match="Unknown color name"):
            get_color_from_latex_string("notacolor!50!white")

    def test_out_of_range_percentage_raises(self):
        with pytest.raises(ValueError, match="Percentages must be in \\[0,100\\]"):
            get_color_from_latex_string("red!101!white")

    def test_empty_expression_raises(self):
        with pytest.raises(ValueError, match="Empty color expression"):
            get_color_from_latex_string("   ")
