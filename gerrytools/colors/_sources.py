"""Named color source registry.

Each public color name (e.g. ``"red"``, ``"tombblue"``, ``"cc:applegreen"``, ``"ensemble:smc"``)
belongs to exactly one *source* — gerrytools' own aliases, the color-blind-friendly cc: palette, the
districtr palette, the LaTeX/CSS table, or matplotlib's named-color mapping. Resolution iterates
these sources in precedence order; a single missing concept ("the source that owns this name") is
what the registry names.

This module is internal; its public functions (`get_named_color`, `which_color_source`,
`get_all_supported_colors_dict`) are re-exported via `gerrytools.colors`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import matplotlib.colors as mcolors

from gerrytools.colors._latex_table import LATEX_COLOR_DICT
from gerrytools.colors.districtr import DISTRICTR_COLOR_DICT
from gerrytools.typing import Color, HexColor

# ---------------------------------------------------------------------------
# Public constants. core.py re-exports these; this module is the single source.
# ---------------------------------------------------------------------------

DEFAULT_GREY = "#5c676f"
"""Default grey plotting color; used in histograms, violin plots, and arrows."""

CITIZEN_BLUE = "#4693b3"
"""Citizen-ensemble blue from the gerrytools palette; resolvable by name as ``"citizen_blue"``."""

OVERLAYS = ("gainsboro", "silver", "darkgray", "gray", "dimgrey")
"""Overlay colors for choropleth maps."""


ENSEMBLE_COLORS = {
    "ensemble:smc": "#ffca5d",
    "ensemble:forest": "#00cd99",
    "ensemble:rrc": "#0099cd",
    "ensemble:revrecom": "#0099cd",
    "ensemble:recom": "#0099cd",
    "ensemble:recoma": "#99cd00",
    "ensemble:recomb": "#cd0099",
    "ensemble:recomc": "#9900cd",
    "ensemble:recomd": "#8dd3c7",
}
"""A dictionary mapping ensemble abbreviations to their corresponding standard colors."""


COLOR_CORRECTED_BASESET = {
    "cc:applegreen": "#73b900",
    "cc:denim": "#0064bd",
    "cc:cherryblossompink": "#ffb0c5",
    "cc:darktangerine": "#ff9f0f",
    "cc:cadmiumgreen": "#006f3c",
    "cc:purpleheart": "#872f9c",
    "cc:alizarin": "#d91b00",
    "cc:greenishcyan": "#009983",
    "cc:lightblue": "#92dbe6",
    "cc:amber": "#ffb900",
    "cc:muddy": "#9b3200",
    "cc:lostinspace": "#003e64",
    "cc:teagreen": "#d0f0c0",
}
"""A small set of colors color-corrected for visibility by color-blind users."""


# The OVERLAYS names resolve through the registry (latexcolors owns all but "dimgrey", which
# falls through to matplotlib) with values identical to matplotlib's; they are deliberately not
# duplicated here.
GERRYTOOLS_EXTRA_COLORS_DICT = {
    "default_grey": DEFAULT_GREY,
    "default_gray": DEFAULT_GREY,
    "citizen_blue": CITIZEN_BLUE,
} | ENSEMBLE_COLORS


# ---------------------------------------------------------------------------
# NamedColorSource: a labelled name → hex mapping with case-insensitive lookup.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NamedColorSource:
    """A labelled name → hex color mapping.

    The ``name`` is for diagnostic provenance (e.g. ``"districtr"``) and is surfaced via
    ``which_color_source``. The ``mapping`` is a dict from color names to hex strings (already
    normalized — no matplotlib named-color references). A lowercased index is built once at
    construction so per-call case-insensitive lookup is free.

    Attributes:
        name (str): Source name included in diagnostic provenance.
        mapping (Mapping[str, str]): Color names mapped to normalized hex strings.
    """

    name: str
    mapping: Mapping[str, str]
    _lowercase_index: Mapping[str, str] = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass to set the cached index.
        object.__setattr__(
            self,
            "_lowercase_index",
            {key.lower(): value for key, value in self.mapping.items()},
        )

    def lookup(self, query: str) -> str | None:
        """Return the hex string for ``query`` if this source defines it.

        Tries the exact form first, then the lowercased form. Returns ``None`` when the source does
        not define ``query`` (which the resolver interprets as "ask the next source in precedence
        order").
        """
        exact_match = self.mapping.get(query)
        if exact_match is not None:
            return exact_match
        return self._lowercase_index.get(query.lower())


# ---------------------------------------------------------------------------
# Pre-built matplotlib hex mapping (snapshotted at import time).
# ---------------------------------------------------------------------------

_MATPLOTLIB_NAMED_AS_HEX: dict[str, str] = {
    color_name: mcolors.to_hex(color_value)
    for color_name, color_value in mcolors.get_named_colors_mapping().items()
}


# ---------------------------------------------------------------------------
# Registry. Order is precedence: earlier sources win.
# ---------------------------------------------------------------------------

# Deliberate: gerrytools resolves "green" to bright #00ff00 (CSS/X11 "lime") instead of matplotlib's
# dark #008000, which reads as forest green in plots. This entry is also the canonical regression
# case for the overrides source outranking matplotlib (see tests/colors/test_sources.py).
_OVERRIDES_SOURCE = NamedColorSource(name="overrides", mapping={"green": "#00ff00"})
_GERRYTOOLS_SOURCE = NamedColorSource(name="gerrytools", mapping=GERRYTOOLS_EXTRA_COLORS_DICT)
_COLOR_CORRECTED_SOURCE = NamedColorSource(name="color-corrected", mapping=COLOR_CORRECTED_BASESET)
_DISTRICTR_SOURCE = NamedColorSource(name="districtr", mapping=DISTRICTR_COLOR_DICT)
_LATEX_SOURCE = NamedColorSource(name="latex", mapping=LATEX_COLOR_DICT)
_MATPLOTLIB_SOURCE = NamedColorSource(name="matplotlib", mapping=_MATPLOTLIB_NAMED_AS_HEX)

# The latex source outranks matplotlib, so a few names both define (e.g. salmon, aquamarine,
# moccasin, lightskyblue) resolve to the latexcolors values. tests/colors/test_sources.py pins
# this precedence; a reorder must be deliberate.
_REGISTRY: tuple[NamedColorSource, ...] = (
    _OVERRIDES_SOURCE,
    _GERRYTOOLS_SOURCE,
    _COLOR_CORRECTED_SOURCE,
    _DISTRICTR_SOURCE,
    _LATEX_SOURCE,
    _MATPLOTLIB_SOURCE,
)


# ---------------------------------------------------------------------------
# Resolution and provenance.
# ---------------------------------------------------------------------------


def get_named_color(query: str) -> HexColor:
    """Resolve a color name through the registry in precedence order.

    Every registry source maps names to hex strings, so the resolved value is always a hex color
    string.

    Args:
        query (str): The name of the color.

    Returns:
        HexColor: The corresponding hex color value.

    Raises:
        KeyError: If no source defines ``query``.
    """
    for source in _REGISTRY:
        hex_value = source.lookup(query)
        if hex_value is not None:
            return hex_value
    raise KeyError(f"Unknown color name: {query!r}")


def which_color_source(query: str) -> str:
    """Return the name of the registry source that owns ``query``.

    Useful for diagnosing precedence: when two palettes both define a name, this answers which one
    the resolver actually returns. Source names currently include ``"overrides"``, ``"gerrytools"``,
    ``"color-corrected"``, ``"districtr"``, ``"latex"``, and ``"matplotlib"``.

    Args:
        query (str): The name of the color.

    Returns:
        str: The name of the source that resolves the color name.

    Raises:
        KeyError: If no source defines ``query``.
    """
    for source in _REGISTRY:
        if source.lookup(query) is not None:
            return source.name
    raise KeyError(f"Unknown color name: {query!r}")


def get_all_supported_colors_dict() -> dict[str, Color]:
    """Get a dictionary of every supported color name mapping to its hex value.

    Composed by walking the registry in *reverse* precedence order so that higher-precedence sources
    overwrite lower-precedence ones — the resulting dict's value for any key matches what
    ``get_named_color`` would return for that key.

    Returns:
        dict[str, Color]: Copy of every supported name mapped to its resolved color.
    """
    composed: dict[str, Color] = {}
    for source in reversed(_REGISTRY):
        composed.update(source.mapping)
    composed["none"] = "none"
    return composed
