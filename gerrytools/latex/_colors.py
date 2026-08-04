"""Shared color parsing helpers for LaTeX output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias, cast

from gerrytools.colors import (
    LATEX_COLOR_DICT,
    convert_color_to_hexa_or_none,
    is_finite_real,
    normalize_rgb_components,
)
from gerrytools.colors.latex import (
    VALID_COLOR_HEX_RE,
    tokenize_xcolor_expression,
)
from gerrytools.colors.latex import normalize_hex_color as _normalize_hex_color_any_length
from gerrytools.typing import Color

# Names containing an apostrophe are excluded: the package defines them with a literal
# backslash-apostrophe (e.g. ``payne\'sgrey``), which cannot be produced by emitting the
# user-facing key verbatim, so those colors must take the HTML fallback path.
_LATEX_COLOR_NAMES = frozenset(
    name
    for name in LATEX_COLOR_DICT
    if name == name.lower() and " " not in name and "'" not in name and name != "classicrose"
)
"""Exact squashed names defined (and safely emittable) by the ``latexcolors`` package."""

LatexColorType: TypeAlias = Literal["NAME", "HTML", "rgb", "RGB"]
LatexColorValue: TypeAlias = str | tuple[float, float, float] | tuple[int, int, int]
LatexColorSpec: TypeAlias = tuple[LatexColorType, LatexColorValue]


def is_hex_color(value: str) -> bool:
    """Check whether a string is a 6-digit hex color code.

    Args:
        value (str): Candidate color string.

    Returns:
        bool: True if ``value`` is ``RRGGBB`` or ``#RRGGBB``.
    """
    stripped = value.strip()
    # The colors-package grammar also admits 3/4/8-digit forms; LaTeX output takes 6 digits only.
    return VALID_COLOR_HEX_RE.match(stripped) is not None and len(stripped.lstrip("#")) == 6


def is_latex_color_expression(value: str) -> bool:
    """Check whether a string is a valid xcolor expression using known LaTeX names.

    Accepted forms are:
    - ``name``
    - ``name!p!name!p!...`` where ``p`` is a percentage in ``[0, 100]``.

    Args:
        value (str): Candidate xcolor expression.

    Returns:
        bool: True if ``value`` parses as a supported xcolor expression and all
            color-name components appear in ``LATEX_COLOR_DICT``.
    """
    try:
        names, _ = tokenize_xcolor_expression(value)
    except ValueError:
        return False

    # Same grammar as gerrytools' resolver, narrower name universe by policy: LaTeX output
    # only preserves expressions whose every name the latexcolors package defines.
    return all(name in _LATEX_COLOR_NAMES for name in names)


def normalize_hex_color(value: str) -> str:
    """Normalize a hex color string.

    Wraps the canonical colors-package normalizer for the 6-digit-only case: the only hex form
    LaTeX ``[HTML]`` arguments accept, with the leading ``#`` dropped.

    Args:
        value (str): Candidate color string.

    Returns:
        str: Lowercase hex color text without a leading ``#``.

    Raises:
        ValueError: If ``value`` is not a valid 6-digit hex color string.
    """
    stripped = value.strip()
    if not is_hex_color(stripped):
        raise ValueError("Color string must be a HEX string in the format '#RRGGBB' or 'RRGGBB'.")
    return _normalize_hex_color_any_length(stripped).lstrip("#")


def to_latex_xcolor_or_html_spec(color: Color) -> LatexColorSpec:
    """Classify a color value for xcolor usage with expression preservation.

    Rules:
    - Valid xcolor expressions (``name`` or ``name!p!name!...``) are preserved as ``("NAME", ...)``.
    - Hex strings are normalized as ``("HTML", "rrggbb")``.
    - Other string colors are converted via GerryTools/Matplotlib parsing and emitted as
      ``("HTML", "rrggbb")``.
    - RGB tuples with all components in ``[0, 1]`` classify as ``"rgb"``; tuples with components
      in ``[0, 255]`` classify as ``"RGB"``. Boolean components and ambiguous tuples (components
      >1 but <=2) are rejected, matching the canonical color model in ``gerrytools.colors``.

    Args:
        color (Color): Color value represented as a name, xcolor expression, hex string,
            or RGB tuple.

    Returns:
        LatexColorSpec: Tuple ``(color_type, color_value)`` suitable for xcolor emitters.

    Raises:
        ValueError: If ``color`` cannot be parsed as a supported color value, or carries a
            non-opaque alpha channel (LaTeX ``[HTML]`` colors cannot represent alpha).
    """
    if isinstance(color, str):
        stripped = color.strip()
        if stripped.lower() == "none":
            raise ValueError("Color 'none' cannot be emitted with \\cellcolor or \\rowcolor.")
        if is_latex_color_expression(stripped):
            return ("NAME", stripped)
        if is_hex_color(stripped):
            return ("HTML", normalize_hex_color(stripped))

        hex8_or_none = convert_color_to_hexa_or_none(stripped)
        hex_digits = hex8_or_none.lstrip("#").lower()
        # A fully-opaque alpha ("ff") truncates losslessly; anything else would be dropped
        # silently, so reject it like the gradient endpoints reject "none".
        if len(hex_digits) == 8 and hex_digits[6:] != "ff":
            raise ValueError(
                f"Color {color!r} carries a non-opaque alpha channel, which LaTeX colors "
                "cannot represent."
            )
        return ("HTML", hex_digits[:6])

    if not isinstance(color, Sequence) or len(color) != 3:
        raise ValueError("Color must be a LaTeX color name, HEX string, or RGB tuple of length 3.")

    # Same component guard as the canonical color model: booleans and non-finite values are not
    # color components even though float() accepts them.
    if not all(is_finite_real(component) for component in color):
        raise ValueError(
            f"RGB components must be finite real numbers; booleans are not valid: {color!r}"
        )

    red = float(color[0])
    green = float(color[1])
    blue = float(color[2])
    if all(0.0 <= component <= 1.0 for component in (red, green, blue)):
        return ("rgb", (red, green, blue))
    if all(0.0 <= component <= 255.0 for component in (red, green, blue)):
        # Canonical-model ambiguity guard: components >1 but <=2 could be either scale.
        normalize_rgb_components(red, green, blue, original_input=color)
        return ("RGB", (int(round(red)), int(round(green)), int(round(blue))))

    raise ValueError("RGB color components must be in the range [0.0, 1.0] or [0, 255].")


TikzColorKind: TypeAlias = Literal["none", "xcolor", "html"]


def classify_tikz_color(color: Color | None) -> tuple[TikzColorKind, str]:
    """Classify a color value for inline TikZ emission.

    Shared front half of the TikZ color handling in the latex plot classes: decides whether a color
    is transparent, a preservable xcolor expression, or needs hex conversion. The plot base class
    emits the result inline (a ``\\color[HTML]{...}`` scope around whole commands, or an extended
    xcolor specification inside option values); no document-level color is registered.

    Args:
        color (Color | None): Color value represented as an xcolor expression, hex string,
            parseable named color, or RGB tuple. None is transparent.

    Returns:
        tuple[TikzColorKind, str]: ``("none", "none")`` for transparent tokens,
            ``("xcolor", expression)`` for valid xcolor expressions, or ``("html", "RRGGBB")`` with
            an uppercase 6-digit hex payload.

    Raises:
        ValueError: If ``color`` cannot be parsed as a supported color value.
    """
    if isinstance(color, str):
        color_expr = color.strip()
        if color_expr.lower() == "none":
            return ("none", "none")
        if is_latex_color_expression(color_expr):
            return ("xcolor", color_expr)

    hex8_or_none = convert_color_to_hexa_or_none(color)
    if hex8_or_none.lower() == "none":
        return ("none", "none")
    hex_digits = hex8_or_none.lstrip("#")
    if len(hex_digits) == 8 and hex_digits[6:].lower() != "ff":
        raise ValueError(
            f"Color {color!r} carries a non-opaque alpha channel, which LaTeX colors "
            "cannot represent."
        )
    return ("html", hex_digits[:6].upper())


def xcolor_args(spec: LatexColorSpec) -> str:
    """Render a :data:`LatexColorSpec` as the argument snippet following an xcolor command.

    The result is ready to append to ``\\cellcolor``/``\\rowcolor``/similar: ``{name}``,
    ``[HTML]{RRGGBB}``, ``[rgb]{r,g,b}``, or ``[RGB]{r,g,b}``.

    Args:
        spec (LatexColorSpec): Classified color mode and normalized value.

    Returns:
        str: Inline xcolor argument snippet.
    """
    color_type, color_value = spec
    if color_type == "NAME":
        return f"{{{color_value}}}"
    if color_type == "HTML":
        return f"[HTML]{{{str(color_value).lstrip('#').upper()}}}"
    if color_type == "rgb":
        red, green, blue = cast("tuple[float, float, float]", color_value)
        return f"[rgb]{{{red:.3f},{green:.3f},{blue:.3f}}}"
    red, green, blue = cast("tuple[int, int, int]", color_value)
    return f"[RGB]{{{red},{green},{blue}}}"


def cellcolor_prefix(color: Color) -> str:
    """Build a ``\\cellcolor`` prefix string.

    Args:
        color (Color): Color value represented as an xcolor expression, hex string,
            RGB tuple, or other parseable color name.

    Returns:
        str: LaTeX snippet like ``\\cellcolor{...}``, ``\\cellcolor[HTML]{...}``, etc.
    """
    return "\\cellcolor" + xcolor_args(to_latex_xcolor_or_html_spec(color))


class CellFillText(str):
    r"""Rendered cell text beginning with a ``\cellcolor`` prefix, carrying its parts.

    The full string is ``\cellcolor<fill_spec><fill_text>``, so plain string consumers
    (``TexTable``) behave exactly as with a plain string. ``TikzTable`` instead reads
    ``fill_spec``/``fill_text`` via :func:`split_cell_fill` and routes the fill into nicematrix's
    ``\CodeBefore``, with no re-parsing of emitted LaTeX. ``compose_formatters`` treats the fill
    as an effect: it peels the spec off after each step, feeds later formatters the plain text,
    and re-attaches the accumulated fill to the final result.

    Attributes:
        fill_spec (str): Inline xcolor argument following ``\cellcolor``, e.g. ``[HTML]{B481D6}``
            or ``{teal}``.
        fill_text (str): Rendered cell text without the ``\cellcolor`` prefix.
    """

    __slots__ = ("fill_spec", "fill_text")

    fill_spec: str
    fill_text: str

    def __new__(cls, fill_spec: str, fill_text: str) -> CellFillText:
        instance = super().__new__(cls, "\\cellcolor" + fill_spec + fill_text)
        instance.fill_spec = fill_spec
        instance.fill_text = fill_text
        return instance


def split_cell_fill(text: str) -> tuple[str | None, str]:
    r"""Split rendered cell text into ``(fill_spec, text_without_prefix)``.

    ``fill_spec`` is an inline xcolor argument ready to follow ``\cellcolor`` or ``\rowcolor``
    (e.g. ``[HTML]{B481D6}`` or ``{teal}``), or ``None`` for plain strings.
    """
    if isinstance(text, CellFillText):
        return text.fill_spec, text.fill_text
    return None, text
