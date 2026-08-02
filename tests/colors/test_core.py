import logging

import pytest

import gerrytools.colors.core as core_module
from gerrytools.colors.core import (
    CITIZEN_BLUE,
    COLOR_CORRECTED_BASESET,
    DEFAULT_GREY,
    ENSEMBLE_COLORS,
    OVERLAYS,
    convert_color_to_hexa_or_none,
    get_all_supported_colors_dict,
    get_named_color,
    resolve_color_and_alpha,
    resolve_rgba,
)

# =====================
# == SUPPORTED NAMES ==
# =====================


class TestSupportedColorDictionaries:
    def test_supported_colors_include_gerrytools_aliases_and_overrides(self):
        supported = get_all_supported_colors_dict()
        assert supported["default_grey"] == DEFAULT_GREY
        assert supported["citizen_blue"] == CITIZEN_BLUE
        assert supported["green"] == "#00ff00"
        assert supported["none"] == "none"

    def test_supported_colors_include_overlay_names(self):
        supported = get_all_supported_colors_dict()
        for overlay in OVERLAYS:
            assert overlay in supported

    def test_color_corrected_baseset_is_a_nonempty_dict(self):
        assert isinstance(COLOR_CORRECTED_BASESET, dict)
        assert len(COLOR_CORRECTED_BASESET) > 0
        assert all(k.startswith("cc:") for k in COLOR_CORRECTED_BASESET)


# =================
# == NAME LOOKUP ==
# =================


class TestNamedColorLookup:
    def test_get_named_color_handles_internal_aliases(self):
        assert get_named_color("citizen_blue") == CITIZEN_BLUE
        assert get_named_color("default_gray") == DEFAULT_GREY

    def test_get_named_color_is_case_insensitive_for_known_palettes(self):
        assert get_named_color("ToMbBlUe") == "#0099cd"

    def test_get_named_color_returns_bright_green_for_green(self):
        # "green" is overridden to bright #00ff00 regardless of source dict
        assert get_named_color("green") == "#00ff00"

    def test_get_named_color_resolves_ensemble_color_names(self):
        # ensemble: prefixed colors live in GERRYTOOLS_EXTRA_COLORS_DICT
        assert get_named_color("ensemble:smc") == ENSEMBLE_COLORS["ensemble:smc"]

    def test_get_named_color_falls_back_to_lowercased_matplotlib_names(self):
        # "tab:blue" is a matplotlib-only name (not in any internal dict);
        # "TAB:BLUE" is not found by exact lookup, so falls back to lowercase.
        expected = get_named_color("tab:blue")
        assert get_named_color("TAB:BLUE") == expected

    def test_get_named_color_unknown_name_raises(self):
        with pytest.raises(KeyError, match="Unknown color name"):
            get_named_color("not-a-real-color")


# ===================
# == HEX CONVERSION ==
# ===================


class TestConvertColorToHexaOrNone:
    def test_convert_none_and_string_none(self):
        assert convert_color_to_hexa_or_none(None) == "none"
        assert convert_color_to_hexa_or_none("none") == "none"

    def test_convert_named_and_latex_colors(self):
        assert convert_color_to_hexa_or_none("green") == "#00ff00ff"
        assert convert_color_to_hexa_or_none("red!50!white") == "#ff8080ff"

    def test_convert_plain_matplotlib_hex8_string(self):
        assert convert_color_to_hexa_or_none("#11223344") == "#11223344"

    def test_convert_rgb_unit_scale_tuple(self):
        assert convert_color_to_hexa_or_none((1.0, 0.5, 0.0)) == "#ff8000ff"

    def test_convert_rgb_255_scale_tuple(self):
        assert convert_color_to_hexa_or_none((255, 128, 0)) == "#ff8000ff"

    def test_convert_rgba_255_scale_with_integer_alpha(self):
        # alpha=64 in 0-255 scale
        assert convert_color_to_hexa_or_none((255, 128, 0, 64)) == "#ff800040"

    def test_convert_rgba_unit_scale_with_float_alpha(self):
        # all four components in [0, 1] — exercises the `else: a = a_raw` branch
        result = convert_color_to_hexa_or_none((0.5, 0.5, 0.5, 0.5))
        assert result.startswith("#")
        assert len(result) == 9  # "#RRGGBBAA"

    def test_convert_base_and_alpha_tuple_with_string_base(self):
        assert convert_color_to_hexa_or_none(("#123456", 0.25)) == "#12345640"

    def test_convert_base_and_alpha_tuple_with_rgb_base(self):
        # base can also be an RGB tuple
        result = convert_color_to_hexa_or_none(((1.0, 0.0, 0.0), 0.5))
        assert result == "#ff000080"

    def test_convert_none_base_with_alpha_tuple_becomes_transparent(self):
        assert convert_color_to_hexa_or_none(("none", 0.25)) == "none"

    # The full validation error matrix lives in test_color_value.py; this representative pair
    # just pins that _Color validation errors surface through the public wrapper.
    def test_convert_ambiguous_rgb_tuple_raises(self):
        with pytest.raises(ValueError, match="Ambiguous RGB tuple"):
            convert_color_to_hexa_or_none((1.5, 0.5, 0.25))

    def test_convert_base_and_alpha_tuple_with_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha in \\(base, alpha\\) color tuple"):
            convert_color_to_hexa_or_none(("#123456", 1.5))

    def test_convert_unknown_string_logs_all_failed_parsers(self, caplog):
        # "mystery-color" naturally fails get_named_color, get_color_from_latex_string,
        # and mcolors.to_rgba — no monkeypatching needed.
        with caplog.at_level(logging.DEBUG, logger=core_module.gt_logger.name):
            with pytest.raises(ValueError, match="Unknown color value"):
                convert_color_to_hexa_or_none("mystery-color")

        assert "not a named color in any gerrytools color source" in caplog.text
        assert "not parsable as a LaTeX color string" in caplog.text
        assert "not parseable by Matplotlib" in caplog.text

    def test_convert_unknown_value_raises(self):
        with pytest.raises(ValueError, match="Unknown color value"):
            convert_color_to_hexa_or_none(object())  # type: ignore


# ==============================
# == RESOLVE COLOR AND ALPHA ==
# ==============================


class TestResolveColorAndAlpha:
    def test_resolve_embedded_alpha_when_no_override_is_given(self):
        color, alpha = resolve_color_and_alpha("#12345680")
        assert color == "#123456"
        assert alpha == pytest.approx(128 / 255)

    def test_resolve_explicit_alpha_overrides_embedded_alpha_and_logs(self, caplog):
        logger = logging.getLogger("tests.colors")
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            color, alpha = resolve_color_and_alpha(
                "#12345680",
                alpha=0.25,
                field="outlinecolor",
                owner="TestOwner",
                logger=logger,
            )

        assert color == "#123456"
        assert alpha == pytest.approx(0.25)
        assert "TestOwner" in caplog.text
        assert "Ignoring alpha embedded in outlinecolor" in caplog.text

    def test_resolve_explicit_alpha_with_no_owner_omits_for_prefix(self, caplog):
        logger = logging.getLogger("tests.colors")
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            color, alpha = resolve_color_and_alpha(
                "#12345680",
                alpha=0.25,
                field="outlinecolor",
                owner=None,
                logger=logger,
            )

        assert color == "#123456"
        assert alpha == pytest.approx(0.25)
        # "For None:" must not appear; just the bare message
        assert "For " not in caplog.text
        assert "Ignoring alpha embedded in outlinecolor" in caplog.text

    def test_resolve_matching_alpha_does_not_log(self, caplog):
        logger = logging.getLogger("tests.colors")
        # "#123456ff" has embedded alpha = 1.0; passing alpha=1.0 should not log
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            color, alpha = resolve_color_and_alpha(
                "#123456ff",
                alpha=1.0,
                field="fillcolor",
                logger=logger,
            )

        assert color == "#123456"
        assert alpha == pytest.approx(1.0)
        assert "Ignoring" not in caplog.text

    def test_resolve_none_returns_zero_alpha_when_allowed(self):
        assert resolve_color_and_alpha("none") == ("none", 0.0)

    def test_resolve_none_with_valid_explicit_alpha_stays_transparent(self):
        assert resolve_color_and_alpha("none", alpha=0.5) == ("none", 0.0)

    def test_resolve_none_still_validates_explicit_alpha(self):
        # Regression: the "none" early return used to skip alpha validation entirely.
        with pytest.raises(ValueError, match=r"fillcolor alpha must be in \[0, 1\]"):
            resolve_color_and_alpha("none", alpha=7.5, field="fillcolor")

    def test_resolve_none_raises_when_not_allowed(self):
        with pytest.raises(ValueError, match="cannot be 'none'"):
            resolve_color_and_alpha("none", allow_none=False, field="fillcolor")

    def test_resolve_invalid_explicit_alpha_raises(self):
        with pytest.raises(ValueError, match=r"fillcolor alpha must be in \[0, 1\]"):
            resolve_color_and_alpha("#123456", alpha=2.0, field="fillcolor")

    def test_resolve_nonfinite_explicit_alpha_raises(self):
        with pytest.raises(ValueError, match="fillcolor alpha must be finite"):
            resolve_color_and_alpha("#123456", alpha=float("nan"), field="fillcolor")


# ==================
# == RESOLVE RGBA ==
# ==================


class TestResolveRgba:
    def test_resolve_rgba_uses_embedded_alpha_without_override(self):
        rgba = resolve_rgba("#12345680")
        assert rgba == pytest.approx((0x12 / 255, 0x34 / 255, 0x56 / 255, 128 / 255))

    def test_resolve_rgba_explicit_alpha_overrides_embedded_alpha(self):
        rgba = resolve_rgba("#12345680", alpha=0.25)
        assert rgba == pytest.approx((0x12 / 255, 0x34 / 255, 0x56 / 255, 0.25))

    def test_resolve_rgba_none_and_python_none_are_fully_transparent(self):
        assert resolve_rgba("none") == (0.0, 0.0, 0.0, 0.0)
        assert resolve_rgba(None) == (0.0, 0.0, 0.0, 0.0)

    def test_resolve_rgba_invalid_alpha_raises_with_field_name(self):
        with pytest.raises(ValueError, match="fillcolor alpha"):
            resolve_rgba("#123456", alpha=2.0, field="fillcolor")
