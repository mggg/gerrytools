import re

from gerrytools.latex._colors import (
    cellcolor_prefix,
    is_hex_color,
    is_latex_color_expression,
)
from gerrytools.typing import Color

# LaTeX control sequence names are letters only;
_CMD_RE = re.compile(r"^[A-Za-z]+$")


def _validate_gradient_colors(*colors: str) -> None:
    """Require color names or expressions that command-based gradients can emit."""
    if any(is_hex_color(color) for color in colors):
        raise ValueError(
            "command-based gradients cannot use hex colors; use command_name=None for HTML colors"
        )
    if not all(is_latex_color_expression(color) for color in colors):
        raise ValueError("command-based gradients require LaTeX color names/expressions")


def validate_command_name(cmd_str: str) -> None:
    """Validate that a LaTeX command name is legal.

    Args:
        cmd_str (str): LaTeX command name without a leading backslash.

    Returns:
        None

    Raises:
        ValueError: If ``cmd_str`` starts with ``"\\\\"`` or contains non-letter characters.
    """
    if cmd_str.startswith("\\"):
        raise ValueError(
            f"`cmd` should not start with '\\\\' (got {cmd_str!r}). Use 'textbf', not '\\\\textbf'."
        )

    if not _CMD_RE.fullmatch(cmd_str):
        raise ValueError(
            f"Illegal LaTeX command name {cmd_str!r}. Please ensure command contains only letters (A-Z, a-z)."
        )


def tex_gradient_command(
    cmd_str: str = "gradient",
    color_name: str = "denim",
    lo: float = 0.0,
    hi: float = 1.0,
    precision: int = 4,
) -> str:
    """Generates a LaTeX command for gradient coloring of numerical values.

    Emits a LaTeX command \\<cmd_str>{x} that colors the cell with a gradient of <color_name> from
    lo (full color) to hi (white). Values outside ``[lo, hi]`` are clamped, and a degenerate range
    (``hi == lo``) is guarded against division by zero, matching the two-color and diverging
    gradient commands.

    Requires: xcolor (with [table]), colortbl, xfp, siunitx.

    Args:
        cmd_str (str, optional): The name of the LaTeX command to create.
            Defaults to ``"gradient"``.
        color_name (str, optional): The name of the color to use for the gradient.
            Defaults to ``"denim"``.
        lo (float, optional): The lower bound of the numerical range. Defaults to ``0.0``.
        hi (float, optional): The upper bound of the numerical range. Defaults to ``1.0``.
        precision (int, optional): Number of decimal places to round the number to.
            Defaults to ``4``.

    Returns:
        str: A string containing the LaTeX command definition.
    """
    # xcolor's ``color!p`` is shorthand for ``color!p!white``, so a single-color gradient is the
    # two-color gradient with a white upper endpoint.
    return tex_twocolor_gradient_command(
        cmd_str,
        lo=lo,
        hi=hi,
        color_lo=color_name,
        color_hi="white",
        precision=precision,
    )


def tex_twocolor_gradient_command(
    cmd_str: str = "heat",
    lo: float = 0.0,
    hi: float = 1.0,
    color_lo: str = "denim",
    color_hi: str = "alizarin",
    precision: int = 4,
) -> str:
    """Generates a LaTeX command for two-color gradient coloring of numerical values.

    Emits a LaTeX command \\<cmd_str>{x} that colors the cell with a two-color gradient from
    color_lo (at lo) to color_hi (at hi), clamped outside the range.

    Requires: xcolor (with [table]), colortbl, xfp, siunitx.

    Note: color mixing is done in xcolor and is a linear interpolation by mixing saturation
    percentages of the two colors.

    Args:
        cmd_str (str, optional): The name of the LaTeX command to create.
            Defaults to ``"heat"``.
        lo (float, optional): The lower bound of the numerical range. Defaults to ``0.0``.
        hi (float, optional): The upper bound of the numerical range. Defaults to ``1.0``.
        color_lo (str, optional): The color at the lower bound. Defaults to ``"denim"``.
        color_hi (str, optional): The color at the upper bound. Defaults to ``"alizarin"``.
        precision (int, optional): Number of decimal places to round the number to.
            Defaults to ``4``.

    Returns:
        str: A string containing the LaTeX command definition to be added to preamble.
    """
    validate_command_name(cmd_str)
    _validate_gradient_colors(color_lo, color_hi)

    return (
        rf"\newcommand{{\{cmd_str}}}[1]{{%"
        "\n"
        r"  \begingroup%"
        "\n"
        rf"  \edef\heatlo{{{lo}}}\edef\heathi{{{hi}}}%"
        "\n"
        r"  \edef\heatrange{\fpeval{max(\heathi-\heatlo, 1e-12)}}%"
        "\n"
        r"  \edef\heatt{\fpeval{min(1, max(0, (#1-\heatlo)/\heatrange))}}%"
        "\n"
        r"  \edef\heatpct{\fpeval{round(100*(1-\heatt),0)}}%"
        "\n"
        rf"  \edef\heatcolorspec{{{color_lo}!\heatpct!{color_hi}}}%"
        "\n"
        r"  \expandafter\cellcolor\expandafter{\heatcolorspec}%"
        "\n"
        rf"  \num[round-mode=places,round-precision={precision}]{{#1}}%"
        "\n"
        r"  \endgroup%"
        "\n"
        r"}"
    )


def tex_cell_highlight_command(cmd_str: str, color: Color = "yellow") -> str:
    """Generates a LaTeX command that applies a cell background color.

    Emits ``\\<cmd_str>{x}``, which renders ``x`` with a ``\\cellcolor`` prefix.

    Args:
        cmd_str (str): The name of the LaTeX command to create.
        color (Color, optional): Cell background color. Defaults to ``"yellow"``.

    Returns:
        str: A string containing the LaTeX command definition.
    """
    validate_command_name(cmd_str)
    prefix = cellcolor_prefix(color)
    return rf"\newcommand{{\{cmd_str}}}[1]{{{prefix}#1}}"


def _tex_ident(s: str) -> str:
    """Convert a string to a valid LaTeX identifier.

    Args:
        s (str): Arbitrary input string.

    Returns:
        str: An identifier containing only letters and digits.
    """
    return re.sub(r"[^A-Za-z0-9]+", "", s) or "X"


def tex_diverging_gradient_command(
    cmd_str: str = "heat",
    lo: float = 0.0,
    mid: float = 0.5,
    hi: float = 1.0,
    color_lo: str = "darkpastelgreen",
    color_hi: str = "richlavender",
    color_mid: str = "white",
    precision: int = 4,
) -> str:
    """Generates a LaTeX command for diverging gradient coloring for numerical values in a table.

    The generated command applies a gradient from ``color_lo`` at ``lo`` through ``color_mid`` at
    ``mid`` to ``color_hi`` at ``hi``. Values are clamped to the interval ``[lo, hi]``.

    Requires: xcolor[table], latexcolor, xfp, siunitx.

    Args:
        cmd_str (str, optional): The name of the LaTeX command to create.
            Defaults to ``"heat"``.
        lo (float, optional): The lower bound of the numerical range. Defaults to ``0.0``.
        mid (float, optional): The midpoint of the numerical range. Defaults to ``0.5``.
        hi (float, optional): The upper bound of the numerical range. Defaults to ``1.0``.
        color_lo (str, optional): The color at the lower bound.
            Defaults to ``"darkpastelgreen"``.
        color_hi (str, optional): The color at the upper bound. Defaults to ``"richlavender"``.
        color_mid (str, optional): The color at the midpoint. Defaults to ``"white"``.
        precision (int, optional): Number of decimal places to round the number to.
            Defaults to ``4``.

    Returns:
        str: A string containing the LaTeX command definition to be added to preamble.
    """
    validate_command_name(cmd_str)
    _validate_gradient_colors(color_lo, color_mid, color_hi)

    lo_name = f"{cmd_str}Lo{_tex_ident(color_lo)}"
    hi_name = f"{cmd_str}Hi{_tex_ident(color_hi)}"
    mid_name = f"{cmd_str}Mid{_tex_ident(color_mid)}"

    return (
        rf"\colorlet{{{lo_name}}}{{{color_lo}}}%"
        "\n"
        rf"\colorlet{{{hi_name}}}{{{color_hi}}}%"
        "\n"
        rf"\colorlet{{{mid_name}}}{{{color_mid}}}%"
        "\n"
        rf"\newcommand{{\{cmd_str}}}[1]{{%"
        "\n"
        r"  \begingroup%"
        "\n"
        rf"  \edef\heatlo{{{lo}}}\edef\heatmid{{{mid}}}\edef\heathi{{{hi}}}%"
        "\n"
        r"  \edef\heatx{#1}%"
        "\n"
        # clamp x into [lo, hi]
        r"  \edef\heatxc{\fpeval{min(\heathi, max(\heatlo, \heatx))}}%"
        "\n"
        # compute left/right ranges safely
        r"  \edef\leftw{\fpeval{max(\heatmid-\heatlo, 1e-12)}}%"
        "\n"
        r"  \edef\rightw{\fpeval{max(\heathi-\heatmid, 1e-12)}}%"
        "\n"
        # choose side; compute pct in 0..100 moving away from the endpoint toward the other
        # endpoint. The comparison stays in fpeval space: an \ifdim on <value>pt overflows TeX's
        # 16383pt dimension ceiling for population-scale ranges.
        r"  \ifnum\fpeval{\heatxc < \heatmid}=1"
        "\n"
        # left: lo -> mid, so pct increases as x approaches mid
        r"    \edef\heatpct{\fpeval{round(100*(1-(\heatxc-\heatlo)/\leftw),0)}}%"
        "\n"
        rf"    \edef\heatcolorspec{{{lo_name}!\heatpct!{mid_name}}}%"
        "\n"
        r"  \else"
        "\n"
        # right: mid -> hi, so pct increases as x approaches hi
        r"    \edef\heatpct{\fpeval{round(100*(1-(\heatxc-\heatmid)/\rightw),0)}}%"
        "\n"
        rf"    \edef\heatcolorspec{{{mid_name}!\heatpct!{hi_name}}}%"
        "\n"
        r"  \fi"
        "\n"
        r"  \expandafter\cellcolor\expandafter{\heatcolorspec}%"
        "\n"
        rf"  \num[round-mode=places,round-precision={precision}]{{#1}}%"
        "\n"
        r"  \endgroup%"
        "\n"
        r"}"
    )
