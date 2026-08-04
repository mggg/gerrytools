"""Tests for the internal canonical `_Color` value.

`_Color.from_any` is the single entry point that absorbs every input shape the
public colors API accepts. These tests exercise that entry point exhaustively;
the public-facing behavior is covered separately in `test_core.py`.
"""

import logging

import pytest

from gerrytools.colors._value import _Color

# =====================
# == NONE / SENTINEL ==
# =====================


class TestColorNoneSentinel:
    def test_from_any_none_input_returns_none_sentinel(self):
        result = _Color.from_any(None)
        assert result.is_none
        assert result.alpha == 0.0
        assert result.to_hex8() == "none"

    def test_from_any_string_none_returns_none_sentinel(self):
        assert _Color.from_any("none").is_none

    def test_from_any_string_none_is_case_insensitive(self):
        assert _Color.from_any("NONE").is_none
        assert _Color.from_any("None").is_none

    def test_classmethod_none_constructs_sentinel(self):
        sentinel = _Color.none()
        assert sentinel.is_none
        assert sentinel.alpha == 0.0


# ====================
# == STRING INPUTS ==
# ====================


class TestColorFromString:
    def test_from_any_named_color_resolves_through_cascade(self):
        # "red" lives in the LATEX_COLOR_DICT; cascade should find it
        red_color = _Color.from_any("red")
        assert red_color.hex6 == "#ff0000"
        assert red_color.alpha == 1.0

    def test_from_any_green_resolves_to_bright_green_override(self):
        # "green" is overridden in the cascade to bright #00ff00
        bright_green = _Color.from_any("green")
        assert bright_green.hex6 == "#00ff00"

    def test_from_any_gerrytools_alias_resolves(self):
        result = _Color.from_any("citizen_blue")
        assert result.hex6 == "#4693b3"

    def test_from_any_latex_xcolor_mix_expression(self):
        # "red!50!white" -> 50% red mixed with 50% white -> #ff8080
        mixed = _Color.from_any("red!50!white")
        assert mixed.hex6 == "#ff8080"

    def test_from_any_plain_hex6_string(self):
        result = _Color.from_any("#11223344")
        assert result.hex6 == "#112233"
        assert result.alpha == pytest.approx(0x44 / 255.0)

    def test_from_any_unknown_string_raises_with_diagnostic(self):
        diagnostic_logger = logging.getLogger("test_color_value.diagnostic")
        with pytest.raises(ValueError, match="Unknown color value"):
            _Color.from_any("not-a-real-color", logger=diagnostic_logger)

    def test_from_any_unknown_string_logs_each_failed_parser(self, caplog):
        diagnostic_logger = logging.getLogger("test_color_value.diagnostic")
        with caplog.at_level(logging.DEBUG, logger=diagnostic_logger.name):
            with pytest.raises(ValueError):
                _Color.from_any("mystery-color", logger=diagnostic_logger)

        assert "not a named color in any gerrytools color source" in caplog.text
        assert "not parsable as a LaTeX color string" in caplog.text
        assert "not parseable by Matplotlib" in caplog.text

    def test_from_any_unknown_string_without_logger_still_raises(self):
        # Diagnostic logger is optional; raising must not depend on it
        with pytest.raises(ValueError, match="Unknown color value"):
            _Color.from_any("definitely-not-a-color")


# ====================
# == TUPLE INPUTS ==
# ====================


class TestColorFromTuples:
    def test_from_any_unit_scale_rgb_tuple(self):
        result = _Color.from_any((1.0, 0.5, 0.0))
        assert result.hex6 == "#ff8000"
        assert result.alpha == 1.0

    def test_from_any_255_scale_rgb_tuple(self):
        result = _Color.from_any((255, 128, 0))
        assert result.hex6 == "#ff8000"
        assert result.alpha == 1.0

    def test_from_any_255_scale_rgba_tuple_with_integer_alpha(self):
        result = _Color.from_any((255, 128, 0, 64))
        assert result.hex6 == "#ff8000"
        assert result.alpha == pytest.approx(64 / 255.0)

    def test_from_any_unit_scale_rgba_tuple(self):
        # alpha round-trips through 8-bit precision so tolerance is ~1/255
        result = _Color.from_any((0.5, 0.5, 0.5, 0.5))
        assert result.alpha == pytest.approx(0.5, abs=1 / 255)

    def test_from_any_base_alpha_pair_with_string_base(self):
        result = _Color.from_any(("#123456", 0.25))
        assert result.hex6 == "#123456"
        assert result.alpha == pytest.approx(0.25, abs=1 / 255)

    def test_from_any_base_alpha_pair_with_rgb_base(self):
        result = _Color.from_any(((1.0, 0.0, 0.0), 0.5))
        assert result.hex6 == "#ff0000"
        assert result.alpha == pytest.approx(0.5, abs=1 / 255)

    def test_from_any_base_alpha_pair_with_none_base_is_none_sentinel(self):
        result = _Color.from_any(("none", 0.25))

        assert result.is_none
        assert result.hex6 == "none"
        assert result.alpha == 0.0


# ============================
# == VALIDATION ERROR PATHS ==
# ============================


class TestColorFromAnyValidationErrors:
    def test_ambiguous_rgb_tuple_raises(self):
        with pytest.raises(ValueError, match="Ambiguous RGB tuple"):
            _Color.from_any((1.5, 0.5, 0.25))

    def test_rgb_tuple_over_255_raises(self):
        with pytest.raises(ValueError, match="<=255"):
            _Color.from_any((256, 0, 0))

    def test_rgb_tuple_negative_in_255_scale_raises(self):
        with pytest.raises(ValueError, match="RGB values must be non-negative"):
            _Color.from_any((-1, 2, 3))

    def test_rgba_tuple_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="Alpha must be non-negative"):
            _Color.from_any((1.0, 0.0, 0.0, -0.1))

    def test_rgba_tuple_alpha_over_255_raises(self):
        with pytest.raises(ValueError, match="Alpha must be <=255"):
            _Color.from_any((255, 0, 0, 256))

    def test_rgba_tuple_ambiguous_alpha_raises(self):
        # Regression: alpha strictly between 1 and 2 used to be silently divided by 255,
        # turning (255, 0, 0, 1.5) into a nearly transparent color.
        with pytest.raises(ValueError, match="Ambiguous alpha"):
            _Color.from_any((255, 0, 0, 1.5))

    def test_rgba_tuple_mixed_scales_raise(self):
        # 0-1 RGB components with a 0-255 alpha is a mixed-scale tuple, not a valid color.
        with pytest.raises(ValueError, match="Mixed-scale RGBA tuple"):
            _Color.from_any((0.5, 0.5, 0.5, 200))

    def test_byte_rgb_rejects_fractional_alpha(self):
        with pytest.raises(ValueError, match="alpha must be an integer"):
            _Color.from_any((200, 100, 50, 0.5))

    def test_rgba_tuple_255_scale_allows_zero_and_one_alpha(self):
        # 0 and 1 are legitimate (nearly transparent) byte values in a 255-scale tuple.
        assert _Color.from_any((255, 0, 0, 0)).alpha == 0.0
        assert _Color.from_any((255, 0, 0, 1)).alpha == pytest.approx(1 / 255.0)

    def test_rgb_tuple_component_of_exactly_two_is_ambiguous(self):
        # The ambiguity boundary is inclusive: exactly 2.0 raises instead of flipping to
        # the 255 scale and producing near-black #020202.
        with pytest.raises(ValueError, match="Ambiguous RGB tuple"):
            _Color.from_any((2.0, 2.0, 2.0))

    def test_base_alpha_pair_with_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha in \\(base, alpha\\) color tuple"):
            _Color.from_any(("#123456", 1.5))

    def test_unknown_value_type_raises(self):
        with pytest.raises(ValueError, match="Unknown color value"):
            _Color.from_any(object())  # type: ignore


# ===========================
# == HEX8 ROUND-TRIPPING ==
# ===========================


class TestColorToHex8:
    def test_to_hex8_for_full_opacity(self):
        opaque_red = _Color(hex6="#ff0000", alpha=1.0)
        assert opaque_red.to_hex8() == "#ff0000ff"

    def test_to_hex8_for_full_transparency(self):
        transparent_red = _Color(hex6="#ff0000", alpha=0.0)
        assert transparent_red.to_hex8() == "#ff000000"

    def test_to_hex8_for_half_opacity(self):
        half_opaque = _Color(hex6="#ff0000", alpha=128 / 255.0)
        assert half_opaque.to_hex8() == "#ff000080"

    def test_to_hex8_for_none_sentinel(self):
        assert _Color.none().to_hex8() == "none"

    def test_round_trip_through_hex8(self):
        # from_any("#11223344") -> _Color -> to_hex8 should produce the same string
        original_hex8 = "#11223344"
        round_tripped = _Color.from_any(original_hex8).to_hex8()
        assert round_tripped == original_hex8


# =====================
# == VALUE SEMANTICS ==
# =====================


class TestColorValueSemantics:
    def test_two_identical_colors_compare_equal(self):
        first_red = _Color(hex6="#ff0000", alpha=1.0)
        second_red = _Color(hex6="#ff0000", alpha=1.0)
        assert first_red == second_red

    def test_color_is_hashable(self):
        opaque_red = _Color(hex6="#ff0000", alpha=1.0)
        opaque_blue = _Color(hex6="#0000ff", alpha=1.0)
        # frozen dataclasses are hashable; this should not raise
        color_set = {opaque_red, opaque_blue, _Color(hex6="#ff0000", alpha=1.0)}
        assert len(color_set) == 2

    def test_color_is_immutable(self):
        opaque_red = _Color(hex6="#ff0000", alpha=1.0)
        with pytest.raises(AttributeError):
            opaque_red.hex6 = "#000000"  # type: ignore
