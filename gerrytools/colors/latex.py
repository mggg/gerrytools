from __future__ import annotations

import re
from warnings import warn

from gerrytools.colors._sources import get_named_color

VALID_COLOR_HEX_RE = re.compile(
    r"^#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$"
)
"""Matches 3/4/6/8-digit hex color strings, with or without a leading ``#``."""


def normalize_hex_color(s: str) -> str:
    """Normalize a hex color string to 6-digit lowercase form "#rrggbb".

    Canonical hex normalizer: accepts 3/4/6/8-digit forms (with or without a leading ``#``),
    expanding shorthand and warning when an alpha channel is dropped.

    Args:
        s (str): A hex color string.

    Returns:
        str: A normalized hex color string in the form "#rrggbb"

    Raises:
        ValueError: If the input string is not a valid hex color.
    """
    s = s.strip()
    if VALID_COLOR_HEX_RE.match(s) is None:
        raise ValueError(f"Not a valid hex color: {s!r}")
    if len(s.strip("#")) == 3:
        s = "#" + "".join(c * 2 for c in s.strip("#"))
    elif len(s.strip("#")) == 4:
        warn("Ignoring alpha channel in hex color: " + s)
        s = "#" + "".join(c * 2 for c in s.strip("#")[:3])
    elif len(s.strip("#")) == 8:
        warn("Ignoring alpha channel in hex color: " + s)
        s = "#" + s.strip("#")[:6]
    elif len(s.strip("#")) == 6:
        s = "#" + s.strip("#")
    return s.lower()


def hex_to_rgb(hex6: str) -> tuple[int, int, int]:
    """Convert a 6-digit hex color string to an RGB tuple of 0-255 integers.

    Args:
        hex6 (str): A hex color string in the form "#RRGGBB" (the leading ``#`` is optional).

    Returns:
        tuple[int, int, int]: A tuple of (R, G, B) values.
    """
    hex6 = normalize_hex_color(hex6)
    return (int(hex6[1:3], 16), int(hex6[3:5], 16), int(hex6[5:7], 16))


def _rgb_to_hex(rgb: tuple[int | float, int | float, int | float]) -> str:
    """Convert an RGB tuple to a 6-digit hex color string.

    Args:
        rgb (tuple[int, int, int]): A tuple of (R, G, B) values.

    Returns:
        str: A hex color string in the form "#RRGGBB"
    """
    r, g, b = rgb
    r = max(0, min(255, int(round(r))))
    g = max(0, min(255, int(round(g))))
    b = max(0, min(255, int(round(b))))
    return f"#{r:02x}{g:02x}{b:02x}"


def _xcolor_mix_hex(hex_colors_list: list[str], percentages_list: list[float | int]) -> str:
    """Allows for mixing of two hex colors according to xcolor semantics.

    See page 44 of the xcolor manual for details:
        https://ctan.math.washington.edu/tex-archive/macros/latex/contrib/xcolor/xcolor.pdf

    Args:
        hex_colors_list (list[str]): A list of hex color strings in the form "#RRGGBB"
        percentages_list (list[float | int]): A list of percentages (0-100) for mixing

    Returns:
        str: The resulting mixed hex color string in the form "#RRGGBB"
    """
    if len(hex_colors_list) != len(percentages_list) + 1:
        raise ValueError(
            "Number of colors must be one more than number of percentages to define the "
            f"interpolation correctly. Found {len(hex_colors_list)} colors and "
            f"{len(percentages_list)} percentages."
        )

    r, g, b = hex_to_rgb(hex_colors_list[0])

    for color, percent in zip(hex_colors_list[1:], percentages_list):
        p = percent / 100.0
        r_mix, g_mix, b_mix = hex_to_rgb(color)
        r = p * r + (1 - p) * r_mix
        g = p * g + (1 - p) * g_mix
        b = p * b + (1 - p) * b_mix
    return _rgb_to_hex((r, g, b))


def tokenize_xcolor_expression(expression: str) -> tuple[list[str], list[float]]:
    """Split an xcolor mix expression into its color-name and percentage tokens.

    The grammar is names at even positions and percentages at odd positions:
    ``"name"``, ``"name!p"``, ``"name!p!other"``, and longer left-folded expressions such as
    ``"name!p!other!q!third"``. An even token count means the expression ends on a percentage,
    which xcolor treats as shorthand for mixing toward white; callers detect that case via
    ``len(names) == len(percents)``.

    Args:
        expression (str): The xcolor expression (e.g., ``"amber!10!denim"``).

    Returns:
        tuple[list[str], list[float]]: The color-name tokens and the percentage values, in order.

    Raises:
        ValueError: If the expression is empty, has an empty segment between ``!`` separators
            (e.g. ``"red!!50"``), or a percentage is non-numeric or outside ``[0, 100]``.
    """
    tokens = [token.strip() for token in expression.strip().split("!")]
    if tokens == [""]:
        raise ValueError("Empty color expression.")
    if any(not token for token in tokens):
        raise ValueError(
            f"Malformed xcolor expression {expression!r}: empty segment between '!' separators."
        )

    names = tokens[::2]
    percents: list[float] = []
    for token in tokens[1::2]:
        try:
            percent = float(token)
        except ValueError:
            raise ValueError(
                f"Malformed xcolor expression {expression!r}: {token!r} is not a percentage."
            ) from None
        if not 0.0 <= percent <= 100.0:
            raise ValueError(f"Percentages must be in [0,100]; got {percent} in {expression!r}.")
        percents.append(percent)
    return names, percents


def get_color_from_latex_string(latex_color_string: str) -> str:
    """Resolve an xcolor-style mix expression into a hex color.

    Supported forms include ``"name"``, ``"name!p"``, ``"name!p!other"``, and longer left-folded
    expressions such as ``"name!p!other!q!third"``. The two-part form ``"name!p"`` is equivalent to
    ``"name!p!white"``.

    Args:
        latex_color_string (str): The xcolor expression (e.g., "amber!10!denim").

    Returns:
        str: The resulting hex color string, in the form "#RRGGBB"
    """
    names, percents = tokenize_xcolor_expression(latex_color_string)
    hex_colors = [normalize_hex_color(get_named_color(name)) for name in names]

    if not percents:
        return hex_colors[0]

    # An even token count ends on a percentage: xcolor's shorthand for mixing toward white.
    if len(hex_colors) == len(percents):
        hex_colors.append("#ffffff")

    return _xcolor_mix_hex(hex_colors, percents)
