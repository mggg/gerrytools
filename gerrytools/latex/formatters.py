from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Hashable
from numbers import Real
from typing import TypeAlias, cast

from gerrytools.colors import convert_color_to_hexa_or_none, hex_to_rgb
from gerrytools.latex._colors import (
    CellFillText,
    split_cell_fill,
    to_latex_xcolor_or_html_spec,
    xcolor_args,
)
from gerrytools.latex.commands import (
    _validate_gradient_colors,
    tex_cell_highlight_command,
    tex_diverging_gradient_command,
    validate_command_name,
)
from gerrytools.typing import Color

TableCellValue: TypeAlias = object
"""Arbitrary DataFrame cell value used by table/formatter pipelines."""

TableIndexValue: TypeAlias = Hashable
"""Hashable index value used by table index-formatting callbacks."""

CellWrapper: TypeAlias = Callable[[TableCellValue, str], tuple[TableCellValue, str]]
"""Formatter callback that receives ``(raw_value, rendered_text)`` and returns updated pair."""

IndexCellWrapper: TypeAlias = Callable[[TableIndexValue, str], tuple[TableIndexValue, str]]
"""Formatter callback for table index values."""

_LATEX_COMMANDS_ATTR = "_gerrytools_latex_commands"


def _digest_suffix(*parts: object, length: int = 6) -> str:
    """Deterministic letters-only suffix from a formatter's canonical construction parameters.

    LaTeX command names may contain only letters, and the digest must be stable across processes,
    so hash the parameter reprs and map digest bytes onto ``a``-``z``. Identical construction
    parameters produce the same command name and body, so registering the same (or an identically
    built) formatter in any number of tables dedupes instead of renaming, and unrelated formatter
    creation order cannot change a table's command names.
    """
    payload = "\x1f".join(repr(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return "".join(chr(ord("a") + byte % 26) for byte in digest[:length])


def latex_commands_for(formatter: Callable) -> tuple[str, ...]:
    """Return LaTeX preamble commands required by a formatter.

    Args:
        formatter (Callable): Formatter callable to inspect.

    Returns:
        tuple[str, ...]: Required LaTeX preamble command definitions.
    """
    return tuple(getattr(formatter, _LATEX_COMMANDS_ATTR, ()))


def _with_latex_commands(formatter: CellWrapper, commands: tuple[str, ...]) -> CellWrapper:
    """Attach required LaTeX preamble commands to a formatter.

    Args:
        formatter (CellWrapper): Formatter to annotate.
        commands (tuple[str, ...]): LaTeX command definitions required by ``formatter``.

    Returns:
        CellWrapper: The same formatter, annotated with command metadata.
    """
    if commands:
        setattr(formatter, _LATEX_COMMANDS_ATTR, commands)
    return formatter


def boxed_center(width: int, height: int | None = None, unit: str = "mm") -> CellWrapper:
    """Create a formatter that centers content in a fixed-size LaTeX ``\\parbox``.

    Args:
        width (int): Box width value.
        height (int | None, optional): Box height value. If None, uses ``width``.
            Defaults to None.
        unit (str, optional): LaTeX unit suffix for ``width``/``height`` (for example ``"mm"``).
            Defaults to ``"mm"``.

    Returns:
        CellWrapper: Formatter that wraps rendered text in a centered parbox.
    """
    if height is None:
        height = width

    def _inner_boxed_center(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        """Format one cell value into a centered fixed-size ``\\parbox``.

        Args:
            v (TableCellValue): Original unformatted cell value.
            s (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Original value and wrapped LaTeX string.
        """
        return v, rf"\parbox[c][{height}{unit}][c]{{{width}{unit}}}{{\centering\strut {s}}}"

    return _inner_boxed_center


def wrap_with_tex_command(cmd_str: str) -> CellWrapper:
    """Wrap rendered cell text in a LaTeX command.

    Args:
        cmd_str (str): LaTeX command name without a leading backslash.

    Returns:
        CellWrapper: Formatter that renders output as ``\\<cmd_str>{...}``.
    """
    validate_command_name(cmd_str)

    def _inner(cell_value: TableCellValue, rendered_str: str) -> tuple[TableCellValue, str]:
        """Wrap one rendered string in the configured TeX command.

        Args:
            cell_value (TableCellValue): Original unformatted cell value.
            rendered_str (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Original value and wrapped LaTeX string.
        """
        return cell_value, rf"\{cmd_str}{{{rendered_str}}}"

    return _inner


def compose_formatters(*funcs: CellWrapper) -> CellWrapper:
    """Compose multiple ``CellWrapper`` formatters into one formatter.

    A cell fill (:class:`CellFillText`) is carried as an effect rather than as text: after each
    step the fill spec is peeled off and accumulated (a later-applied fill overrides an earlier
    one), subsequent formatters see the plain rendered text, and the accumulated fill is
    re-attached to the final result. A wrapper composed around a fill formatter therefore wraps
    the text only, and the fill still reaches the table emitters.

    Args:
        *funcs (CellWrapper): One or more formatter callables.

    Returns:
        CellWrapper: A formatter that applies all provided formatters from right to left.
    """

    # compose(f, g, h)(v, s) == f(g(h(v, s)))
    def run(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        """Run composed formatters over a single value/string pair.

        Args:
            v (TableCellValue): Original unformatted cell value.
            s (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Updated value/string pair after all formatters.
        """
        fill_spec: str | None = None
        for formatter in reversed(funcs):
            v, s = formatter(v, s)
            step_fill, s = split_cell_fill(s)
            if step_fill is not None:
                fill_spec = step_fill
        if fill_spec is not None:
            return v, CellFillText(fill_spec, s)
        return v, s

    commands: list[str] = []
    for formatter in funcs:
        commands.extend(latex_commands_for(formatter))

    return _with_latex_commands(run, tuple(commands))


def round_decimals(decimal_places: int) -> CellWrapper:
    """Create a formatter that renders numeric values with fixed decimal places.

    Args:
        decimal_places (int): Number of decimal places to render.

    Returns:
        CellWrapper: Formatter that applies fixed-point rendering to numeric values.
    """

    def _inner_round_decimals(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        """Render one numeric value with the configured decimal precision.

        Args:
            v (TableCellValue): Original unformatted cell value.
            s (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Original value and precision-formatted output string.
        """
        if isinstance(v, Real):
            rendered = f"{v:.{decimal_places}f}"
            if rendered.startswith("-") and float(rendered) == 0.0:
                rendered = rendered[1:]  # normalize "-0.00" to "0.00"
            return v, rendered
        return v, s

    return _inner_round_decimals


def _safe_round(v: TableCellValue, round_to: int | None) -> TableCellValue:
    """Round numeric values while preserving ``NaN`` and infinite values.

    Args:
        v (TableCellValue): Candidate value.
        round_to (int | None): Decimal places to round to, if provided.

    Returns:
        TableCellValue: Rounded numeric value or the original input.
    """
    if isinstance(v, Real) and round_to is not None:
        if v != v:  # NaN check
            return v
        if v == float("inf"):
            return v
        if v == float("-inf"):
            return v
        return float(round(v, round_to))
    return v


def _make_numeric_highlighter(
    predicate: Callable[[float], bool],
    color: Color,
    *,
    round_to: int | None,
    command_prefix: str | None,
    kind: str,
    thresholds: tuple[float, ...],
) -> CellWrapper:
    """Build a numeric highlighter formatter from a predicate.

    Args:
        predicate (Callable[[float], bool]): Comparison predicate.
        color (Color): Highlight color.
        round_to (int | None): Decimal places to round values before comparison.
        command_prefix (str | None): Command-name prefix for compact command
            output. Pass ``None`` to emit literal ``\\cellcolor`` prefixes.
        kind (str): Comparison kind tag (``"gt"``, ``"le"``, ...) for the command-name digest.
        thresholds (tuple[float, ...]): Effective threshold values for the command-name digest.

    Returns:
        CellWrapper: Formatter that prepends a LaTeX ``\\cellcolor`` command when matched.
    """
    if command_prefix is not None:
        validate_command_name(command_prefix)
        selected_name = command_prefix + _digest_suffix(
            kind, thresholds, round_to, color, command_prefix
        )
        command = tex_cell_highlight_command(selected_name, color)

        def _inner_command(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
            if isinstance(v, Real):
                rounded_value = _safe_round(v, round_to)
                if isinstance(rounded_value, Real) and predicate(float(rounded_value)):
                    return v, rf"\{selected_name}{{{s}}}"
            return v, s

        return _with_latex_commands(_inner_command, (command,))

    fill_spec = xcolor_args(to_latex_xcolor_or_html_spec(color))

    def _inner_literal(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        """Apply conditional cell highlighting to one value/string pair.

        Args:
            v (TableCellValue): Original unformatted cell value.
            s (str): Current rendered cell string.

        Returns:
            tuple[TableCellValue, str]: Original value and highlighted (or unchanged) output string.
        """
        if isinstance(v, Real):
            rounded_value = _safe_round(v, round_to)
            if isinstance(rounded_value, Real) and predicate(float(rounded_value)):
                return v, CellFillText(fill_spec, s)
        return v, s

    return _inner_literal


def _between_bounds(
    lower_bound: int | float,
    upper_bound: int | float,
    include_lower: bool,
    include_upper: bool,
) -> tuple[float, float]:
    """Resolve inclusive/exclusive bounds into a closed float interval.

    Exclusive bounds are nudged one float toward the interior with ``math.nextafter``, so the
    predicate stays a plain ``low <= x <= high``.
    """
    low = float(lower_bound)
    high = float(upper_bound)
    if not include_lower:
        low = math.nextafter(low, math.inf)  # smallest float > lower_bound
    if not include_upper:
        high = math.nextafter(high, -math.inf)  # largest float < upper_bound
    return low, high


def highlight_gt(
    thresh: int | float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    command_prefix: str | None = None,
) -> CellWrapper:
    """Highlight values strictly greater than a threshold.

    Args:
        thresh (int | float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        command_prefix (str | None, optional): Prefix for generated compact
            LaTeX commands. Pass ``None`` for literal ``\\cellcolor`` output.
            Defaults to None.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(
        lambda x: x > float(thresh),
        color,
        round_to=round_to,
        command_prefix=command_prefix,
        kind="gt",
        thresholds=(float(thresh),),
    )


def highlight_ge(
    thresh: int | float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    command_prefix: str | None = None,
) -> CellWrapper:
    """Highlight values greater than or equal to a threshold.

    Args:
        thresh (int | float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        command_prefix (str | None, optional): Prefix for generated compact
            LaTeX commands. Pass ``None`` for literal ``\\cellcolor`` output.
            Defaults to None.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(
        lambda x: x >= float(thresh),
        color,
        round_to=round_to,
        command_prefix=command_prefix,
        kind="ge",
        thresholds=(float(thresh),),
    )


def highlight_lt(
    thresh: float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    command_prefix: str | None = None,
) -> CellWrapper:
    """Highlight values strictly less than a threshold.

    Args:
        thresh (float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        command_prefix (str | None, optional): Prefix for generated compact
            LaTeX commands. Pass ``None`` for literal ``\\cellcolor`` output.
            Defaults to None.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(
        lambda x: x < float(thresh),
        color,
        round_to=round_to,
        command_prefix=command_prefix,
        kind="lt",
        thresholds=(float(thresh),),
    )


def highlight_le(
    thresh: float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    command_prefix: str | None = None,
) -> CellWrapper:
    """Highlight values less than or equal to a threshold.

    Args:
        thresh (float): Threshold value.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        command_prefix (str | None, optional): Prefix for generated compact
            LaTeX commands. Pass ``None`` for literal ``\\cellcolor`` output.
            Defaults to None.

    Returns:
        CellWrapper: Highlight formatter.
    """
    return _make_numeric_highlighter(
        lambda x: x <= float(thresh),
        color,
        round_to=round_to,
        command_prefix=command_prefix,
        kind="le",
        thresholds=(float(thresh),),
    )


def highlight_between(
    lower_bound: int | float,
    upper_bound: int | float,
    color: Color = "yellow",
    *,
    round_to: int | None = None,
    include_lower: bool = True,
    include_upper: bool = True,
    command_prefix: str | None = None,
) -> CellWrapper:
    """Highlight values between lower and upper bounds.

    Args:
        lower_bound (int | float): Lower bound.
        upper_bound (int | float): Upper bound.
        color (Color, optional): Highlight color. Defaults to ``"yellow"``.
        round_to (int | None, optional): Decimal places for comparison rounding.
            Defaults to None.
        include_lower (bool, optional): Whether the lower bound is inclusive.
            Defaults to True.
        include_upper (bool, optional): Whether the upper bound is inclusive.
            Defaults to True.
        command_prefix (str | None, optional): Prefix for generated compact
            LaTeX commands. Pass ``None`` for literal ``\\cellcolor`` output.
            Defaults to None.

    Returns:
        CellWrapper: Highlight formatter.
    """
    low, high = _between_bounds(lower_bound, upper_bound, include_lower, include_upper)
    return _make_numeric_highlighter(
        lambda x: low <= x <= high,
        color,
        round_to=round_to,
        command_prefix=command_prefix,
        kind="btw",
        thresholds=(low, high),
    )


def diverging_gradient_formatter(
    lo: float = 0.0,
    mid: float = 0.5,
    hi: float = 1.0,
    color_lo: Color = "darkpastelgreen",
    color_hi: Color = "richlavender",
    color_mid: Color = "white",
    *,
    command_name: str | None = "divgrad",
    precision: int = 4,
) -> CellWrapper:
    """Formatter that applies a diverging gradient cell background.

    By default, renders numeric cells as compact LaTeX command calls like ``\\divgrad{0.774}`` and
    exposes the matching preamble command so ``TexTable`` can add it automatically when the
    formatter is set.

    Pass ``command_name=None`` to use the literal-color path instead.  That computes the
    interpolated background color in Python and prepends a ``\\cellcolor[HTML]{RRGGBB}`` to the
    rendered string.  This is the preferred path for ``TikzTable``, which routes the carried fill
    spec into nicematrix's ``\\CodeBefore`` so the fill spans the full cell.

    Args:
        lo (float): Lower bound of the gradient range. Defaults to ``0.0``.
        mid (float): Midpoint of the gradient range. Defaults to ``0.5``.
        hi (float): Upper bound of the gradient range. Defaults to ``1.0``.
        color_lo (Color): Color at the lower bound. Defaults to ``"darkpastelgreen"``.
        color_hi (Color): Color at the upper bound. Defaults to ``"richlavender"``.
        color_mid (Color): Color at the midpoint. Defaults to ``"white"``.
        command_name (str | None): LaTeX command name for compact command-based
            output. Pass ``None`` for literal ``\\cellcolor[HTML]{...}``
            prefixes. Defaults to ``"divgrad"``.
        precision (int): ``siunitx`` round precision used by the generated
            command when ``command_name`` is provided. Defaults to ``4``.

    Returns:
        CellWrapper: Formatter that applies gradient coloring to numeric cells.

    Raises:
        ValueError: If bounds, precision, a command name, or command-based colors are invalid.
    """
    if command_name is not None:
        if not all(isinstance(c, str) for c in (color_lo, color_mid, color_hi)):
            raise ValueError("command-based gradients require LaTeX color names/expressions")
        color_lo_str = cast(str, color_lo)
        color_mid_str = cast(str, color_mid)
        color_hi_str = cast(str, color_hi)
        _validate_gradient_colors(color_lo_str, color_mid_str, color_hi_str)
        validate_command_name(command_name)
        selected_name = command_name + _digest_suffix(
            "divgrad",
            lo,
            mid,
            hi,
            color_lo_str,
            color_mid_str,
            color_hi_str,
            precision,
            command_name,
        )
        command = tex_diverging_gradient_command(
            selected_name,
            lo=lo,
            mid=mid,
            hi=hi,
            color_lo=color_lo_str,
            color_mid=color_mid_str,
            color_hi=color_hi_str,
            precision=precision,
        )

        def _inner_command(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
            if not isinstance(v, Real):
                return v, s
            value = float(v)
            if math.isinf(value):
                # \command{inf} is a siunitx error; clamp to the endpoint color like the
                # literal path and keep the rendered text.
                endpoint_color = color_hi_str if value > 0 else color_lo_str
                return v, CellFillText(f"{{{endpoint_color}}}", s)
            return v, rf"\{selected_name}{{{value:.{precision}f}}}"

        return _with_latex_commands(_inner_command, (command,))

    def _resolve(c: Color) -> tuple[int, int, int]:
        hex_str = convert_color_to_hexa_or_none(c)
        if hex_str.lower() == "none":
            raise ValueError(
                f"Gradient endpoint colors cannot be 'none' (got {c!r}); "
                "a transparent endpoint cannot be interpolated."
            )
        # Truncate the hexa alpha byte before parsing; the endpoints are opaque by contract.
        return hex_to_rgb(hex_str.lstrip("#")[:6])

    rgb_lo = _resolve(color_lo)
    rgb_mid = _resolve(color_mid)
    rgb_hi = _resolve(color_hi)
    lo_f, mid_f, hi_f = float(lo), float(mid), float(hi)
    left_w = max(mid_f - lo_f, 1e-12)
    right_w = max(hi_f - mid_f, 1e-12)

    def _lerp(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
        t = max(0.0, min(1.0, t))
        return (
            round(c1[0] + (c2[0] - c1[0]) * t),
            round(c1[1] + (c2[1] - c1[1]) * t),
            round(c1[2] + (c2[2] - c1[2]) * t),
        )

    def _inner(v: TableCellValue, s: str) -> tuple[TableCellValue, str]:
        if not isinstance(v, Real):
            return v, s
        x = float(v)
        x = max(lo_f, min(hi_f, x))
        if x < mid_f:
            rgb = _lerp(rgb_lo, rgb_mid, (x - lo_f) / left_w)
        else:
            rgb = _lerp(rgb_mid, rgb_hi, (x - mid_f) / right_w)
        hex_color = f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        return v, CellFillText(f"[HTML]{{{hex_color}}}", s)

    return _inner
