from string import hexdigits

import pytest

from gerrytools.colors.districtr import DISTRICTR_COLOR_DICT, districtr, hexshift

# ==================
# == HEX SHIFTING ==
# ==================


class TestHexshift:
    def test_hexshift_is_deterministic_for_a_seed(self):
        shifted_once = hexshift("#ffca5d", seed=7)
        shifted_twice = hexshift("#ffca5d", seed=7)
        assert shifted_once == shifted_twice

    def test_hexshift_preserves_hex_shape_but_changes_value(self):
        shifted = hexshift("#ffca5d", seed=11)
        assert shifted.startswith("#")
        assert len(shifted) == 7
        assert all(char in hexdigits for char in shifted[1:])
        assert shifted != "#ffca5d"

    def test_hexshift_produces_different_result_for_different_seeds(self):
        a = hexshift("#ffca5d", seed=1)
        b = hexshift("#ffca5d", seed=2)
        # Different seeds should (almost certainly) produce different outputs
        assert a != b

    def test_hexshift_uppercase_input_changes_the_color_value(self):
        # Regression: an uppercase input could pass the inequality guard with a case-only
        # change (e.g. "#FFCA5D" -> "#ffCA5D"), returning the same color value.
        for seed in range(25):
            shifted = hexshift("#FFCA5D", seed=seed)
            assert shifted == shifted.lower()
            assert shifted != "#ffca5d"

    def test_hexshift_normalizes_missing_hash_prefix(self):
        # Regression: without the leading "#", the digit selection was off by one and the
        # result lacked the prefix.
        assert hexshift("ffca5d", seed=11) == hexshift("#ffca5d", seed=11)

    @pytest.mark.parametrize("bad_color", ["red", "#fff", "#ffca5d00", "", 0xFFCA5D, None])
    def test_hexshift_rejects_non_hex_input(self, bad_color):
        # Regression: hexshift("red") happily returned "r3d".
        with pytest.raises(ValueError, match="six-digit hex color"):
            hexshift(bad_color)


# =======================
# == DISTRICTR PALETTE ==
# =======================


class TestDistrictrPalette:
    def test_districtr_returns_requested_prefix_for_small_n(self):
        base_colors = list(DISTRICTR_COLOR_DICT.values())
        assert districtr(3) == base_colors[:3]

    def test_districtr_returns_empty_list_for_zero(self):
        assert districtr(0) == []

    def test_districtr_rejects_negative_size(self):
        with pytest.raises(ValueError, match="nonnegative"):
            districtr(-1)

    def test_districtr_returns_full_base_palette_at_exact_boundary(self):
        # N == len(base) — tail is empty, no hexshifting occurs
        base_colors = list(DISTRICTR_COLOR_DICT.values())
        result = districtr(len(base_colors))
        assert result == base_colors

    def test_districtr_extends_palette_with_shifted_tail(self):
        base_colors = list(DISTRICTR_COLOR_DICT.values())
        extended = districtr(len(base_colors) + 2)
        assert len(extended) == len(base_colors) + 2
        assert extended[: len(base_colors)] == base_colors
        assert extended[len(base_colors)] != base_colors[0]
        assert extended[len(base_colors) + 1] != base_colors[1]

    def test_districtr_extension_rounds_do_not_repeat(self):
        n_colors = 3 * len(DISTRICTR_COLOR_DICT) + 3
        palette = districtr(n_colors)
        assert len(palette) == n_colors
        assert len(set(palette)) == n_colors

    def test_districtr_extension_is_deterministic(self):
        assert districtr(200) == districtr(200)


def test_districtr_raises_instead_of_padding_with_duplicates(monkeypatch):
    # The exhaustion escape hatch used to append silent duplicates. The module is fetched via
    # importlib because the package re-exports the districtr *function* under the same name.
    import importlib

    districtr_module = importlib.import_module("gerrytools.colors.districtr")
    monkeypatch.setattr(districtr_module, "hexshift", lambda color, *, seed: "#0099cd")
    n = len(districtr_module.DISTRICTR_COLOR_DICT) + 1
    with pytest.raises(RuntimeError, match="distinct districtr colors"):
        districtr_module.districtr(n)
