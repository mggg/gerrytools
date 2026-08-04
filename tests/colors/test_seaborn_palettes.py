import pytest

import gerrytools.colors.seaborn as seaborn_palettes
from gerrytools.colors.seaborn import flare, greenpurplecmap, greens, purples, redbluecmap


def _all_triples_are_unit_rgb(colors):
    return all(
        len(color) == 3 and all(0.0 <= channel <= 1.0 for channel in color) for color in colors
    )


# ======================
# == DIVERGING MAPS ==
# ======================


class TestDivergingPaletteHelpers:
    def test_redbluecmap_returns_requested_number_of_colors(self):
        colors = redbluecmap(4)
        assert len(colors) == 4
        assert _all_triples_are_unit_rgb(colors)

    def test_redbluecmap_rejects_non_positive_sizes(self):
        with pytest.raises(ValueError, match="positive integer"):
            redbluecmap(0)

    def test_greenpurplecmap_whitens_the_middle_of_odd_palettes(self):
        colors = greenpurplecmap(5)
        assert len(colors) == 5
        assert colors[2] == pytest.approx((240 / 255, 240 / 255, 240 / 255))

    def test_greenpurplecmap_single_color_is_whitened(self):
        # n=1 is odd; mid=0, so the single color is replaced with near-white
        colors = greenpurplecmap(1)
        assert len(colors) == 1
        assert colors[0] == pytest.approx((240 / 255, 240 / 255, 240 / 255))

    def test_greenpurplecmap_even_n_skips_whitening(self):
        # n=4 is even; the middle-whitening branch is not entered
        colors = greenpurplecmap(4)
        assert len(colors) == 4
        assert _all_triples_are_unit_rgb(colors)
        # No element should be the whitened colour
        white = (240 / 255, 240 / 255, 240 / 255)
        assert not any(c == pytest.approx(white) for c in colors)

    def test_greenpurplecmap_rejects_non_positive_sizes(self):
        with pytest.raises(ValueError, match="positive integer"):
            greenpurplecmap(-1)


# ====================
# == SEQUENTIAL MAPS ==
# ====================


class TestSequentialPaletteHelpers:
    @pytest.mark.parametrize(
        ("helper", "palette_name"),
        [(flare, "flare"), (purples, "Purples"), (greens, "Greens")],
    )
    def test_helpers_select_their_named_palette(self, monkeypatch, helper, palette_name):
        calls = []

        def fake_palette(name, n, *, reverse=False):
            calls.append((name, n, reverse))
            return [(0.1, 0.2, 0.3)] * n

        monkeypatch.setattr(seaborn_palettes, "_seaborn_palette", fake_palette)

        assert helper(3) == [(0.1, 0.2, 0.3)] * 3
        assert calls == [(palette_name, 3, False)]

    def test_flare_returns_unit_rgb_triples(self):
        colors = flare(3)
        assert len(colors) == 3
        assert _all_triples_are_unit_rgb(colors)

    def test_purples_returns_unit_rgb_triples(self):
        colors = purples(3)
        assert len(colors) == 3
        assert _all_triples_are_unit_rgb(colors)

    def test_greens_returns_unit_rgb_triples(self):
        colors = greens(3)
        assert len(colors) == 3
        assert _all_triples_are_unit_rgb(colors)
