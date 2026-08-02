from collections.abc import Mapping, Sequence
from typing import Union, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from gerrytools.colors.core import resolve_rgba
from gerrytools.typing import Color


def _resolve_rgb_triple(color: Color) -> tuple[float, float, float]:
    """Resolve any gerrytools color input to an opaque RGB triple for swatch fills."""
    red, green, blue, _alpha = resolve_rgba(color)
    return (red, green, blue)


def _swatch_text_color(rgb: tuple[float, float, float]) -> str:
    """Black or white, whichever contrasts with a swatch of the given fill.

    Uses the Rec. 601 luma weights rather than a raw channel sum, so a saturated
    green swatch (bright to the eye) gets black text while a saturated blue one
    (dark to the eye) gets white.
    """
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.5 else "white"


def _annotate_swatch(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    rgb: tuple[float, float, float],
    *,
    fontsize: float,
) -> None:
    """Write centered annotation text on one swatch in a contrasting color."""
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=_swatch_text_color(rgb),
    )


def preview_palette(
    colors: Sequence[Color],
    figsize: tuple[float, float] = (6, 1),
    show_indices: bool = False,
    show_hex: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Preview a color palette as horizontal swatches.

    Args:
        colors (Sequence[Color]): Sequence of colors. Each can be anything the gerrytools resolver
            accepts: a package color name (e.g. "citizen_blue"), a hex string (e.g. "#ff0000"), or
            an RGB triple.
        figsize (tuple[float, float]): Size of the figure (width, height). Ignored when ``ax`` is
            provided.
        show_indices (bool): If True, annotate each swatch with its index.
        show_hex (bool): If True, annotate each swatch with its hex code.
        ax (Axes | None): Optional existing axes to draw onto. If None, a fresh figure and axes are
            created.

    Returns:
        tuple[Figure, Axes]: The Matplotlib figure and axes.

    Raises:
        ValueError: If ``colors`` is empty or a color cannot be resolved.
    """
    if len(colors) == 0:
        raise ValueError("No colors provided.")

    rgb_colors = [_resolve_rgb_triple(c) for c in colors]
    n = len(rgb_colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = cast(Figure, ax.figure)

    # Draw swatches as vertical bars across the axis
    for i, c in enumerate(rgb_colors):
        ax.bar(
            i,
            1,
            color=c,
            edgecolor="none",
            align="edge",
            width=1.0,
        )

        if show_indices or show_hex:
            text = ""
            if show_indices:
                text += str(i)
            if show_hex:
                if text:
                    text += "\n"
                text += mcolors.to_hex(c)
            _annotate_swatch(ax, i + 0.5, 0.5, text, c, fontsize=8)

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig, ax


def compare_palettes(
    palettes: Union[Mapping[str, Sequence[Color]], Sequence[Sequence[Color]]],
    figsize: tuple[float, float] | None = None,
    show_hex: bool = False,
) -> tuple[Figure, Axes]:
    """
    Compare multiple color palettes as horizontal rows.

    Args:
        palettes (Union[Mapping[str, Sequence[Color]], Sequence[Sequence[Color]]]): A mapping from
            palette names to color sequences, or a sequence of color sequences whose rows will be
            named 0, 1, 2, and so on. Colors can be anything the gerrytools resolver accepts:
            package color names, hex strings, or RGB triples.
        figsize (tuple[float, float] | None, optional): Matplotlib figure size. Defaults to None,
            which chooses dimensions from the number and maximum length of the palettes.
        show_hex (bool, optional): Whether to write hex codes inside swatches. Defaults to False.

    Returns:
        tuple[Figure, Axes]: (fig, ax)

    Raises:
        ValueError: If ``palettes`` is empty or contains an invalid color.
    """
    # Normalize palettes to dict[name -> list[RGB triples]]
    items: list[tuple[str, Sequence[Color]]]
    if isinstance(palettes, Mapping):
        # Cast to satisfy the type-checker after runtime Mapping narrowing.
        palette_map = cast(Mapping[str, Sequence[Color]], palettes)
        items = [(str(name), palette) for name, palette in palette_map.items()]
    else:
        items = [(str(i), row) for i, row in enumerate(palettes)]

    norm_palettes: list[tuple[str, list[tuple[float, float, float]]]] = []
    max_len = 0
    for name, colors in items:
        rgb_colors = [_resolve_rgb_triple(c) for c in colors]
        norm_palettes.append((name, rgb_colors))
        max_len = max(max_len, len(rgb_colors))

    n_palettes = len(norm_palettes)
    if n_palettes == 0:
        raise ValueError("No palettes provided.")

    # Default figsize: scale with number of colors and palettes
    if figsize is None:
        figsize = (max(4, max_len * 0.5), max(1.5, n_palettes * 0.6))

    fig, ax = plt.subplots(figsize=figsize)

    # Draw each palette as a row (y = row index)
    for row_idx, (name, colors) in enumerate(norm_palettes):
        y = n_palettes - 1 - row_idx  # invert so first is at top
        for col_idx, c in enumerate(colors):
            rect = Rectangle((col_idx, y), 1, 1, facecolor=c, edgecolor="none")
            ax.add_patch(rect)

            if show_hex:
                _annotate_swatch(ax, col_idx + 0.5, y + 0.5, mcolors.to_hex(c), c, fontsize=6)

        # Palette label on the left
        ax.text(
            -0.3,
            y + 0.5,
            name,
            ha="right",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max_len)
    ax.set_ylim(0, n_palettes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig, ax
