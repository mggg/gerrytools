"""Shared machinery for the TikZ plot classes (TikzSeatsVotesPlot, TikzPaintballPlot).

One base class owns the document plumbing, the axis/scale setter surface, and the TikZ command
builders, so the two plot dialects cannot drift. Colors follow one strategy: tokens classified by
:func:`gerrytools.latex._colors.classify_tikz_color` and emitted inline (a ``\\color[HTML]{...}``
scope around whole commands, or an extended xcolor specification inside option values), never by
mutating the document's color table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, get_args

from gerrytools.colors.latex import hex_to_rgb
from gerrytools.latex._colors import TikzColorKind, classify_tikz_color
from gerrytools.latex.document import TexDocument

# Re-exported so latex callers keep importing the sentinel from here; the single definition lives
# in the dependency-light typing hub and is shared with gerrytools.plotting.
from gerrytools.typing import UNSET as UNSET
from gerrytools.typing import Color, TikzLineStyle
from gerrytools.typing import Unset as Unset


def _to_tikz_linestyle(linestyle: str) -> str:
    """Map Matplotlib-style line strings to TikZ line styles.

    Args:
        linestyle (str): Matplotlib-style or TikZ-style line token.

    Returns:
        str: Equivalent TikZ line style token.

    Raises:
        ValueError: If ``linestyle`` is neither a known Matplotlib token nor a valid TikZ line
            style. Unknown tokens used to pass through silently and break the LaTeX compile instead.
    """
    style_map = {
        "-": "solid",
        "--": "dashed",
        ":": "dotted",
        "-.": "dashdotted",
        "dashdot": "dashdotted",
    }
    mapped_style = style_map.get(str(linestyle), str(linestyle))
    valid_linestyles = get_args(TikzLineStyle)
    if mapped_style not in valid_linestyles:
        raise ValueError(
            f"Invalid linestyle: {linestyle!r}. Must be a Matplotlib token "
            f"({', '.join(repr(token) for token in style_map)}) or a TikZ line style "
            f"({', '.join(repr(style) for style in valid_linestyles)})."
        )
    return mapped_style


@dataclass(frozen=True)
class _GuideLine:
    """Validated guide line shared by the TikZ plot classes."""

    slope: float
    linecolor: Color | None
    linewidth: float
    linestyle: str
    label: str | None = None

    def __post_init__(self) -> None:
        slope = float(self.slope)
        if math.isnan(slope):
            raise ValueError("slope must not be NaN.")
        object.__setattr__(self, "slope", slope)

        linewidth = float(self.linewidth)
        if not math.isfinite(linewidth):
            raise ValueError("linewidth must be finite.")
        if linewidth < 0:
            raise ValueError("linewidth must be nonnegative.")
        object.__setattr__(self, "linewidth", linewidth)
        object.__setattr__(self, "linestyle", _to_tikz_linestyle(str(self.linestyle)))


# ---------------------------------------------------------------------------
# Option validation
# ---------------------------------------------------------------------------

OptionValidator = Callable[[Any], object]


def _rounded(number: float, round_to: int | None) -> float:
    return number if round_to is None else round(number, round_to)


def passthrough_option(value: Any) -> object:
    """Accept any value unchanged (colors and other free-form options)."""
    return value


def positive_float_option(key: str, *, round_to: int | None = None) -> OptionValidator:
    """Validator for a finite, strictly positive float option."""

    def validate(value: Any) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{key} must be finite.")
        if number <= 0:
            raise ValueError(f"{key} must be positive.")
        return _rounded(number, round_to)

    return validate


def nonnegative_float_option(key: str, *, round_to: int | None = None) -> OptionValidator:
    """Validator for a finite, nonnegative float option."""

    def validate(value: Any) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{key} must be finite.")
        if number < 0:
            raise ValueError(f"{key} must be nonnegative.")
        return _rounded(number, round_to)

    return validate


def unit_interval_option(key: str, *, round_to: int | None = None) -> OptionValidator:
    """Validator for a float option constrained to ``[0, 1]``."""

    def validate(value: Any) -> float:
        number = float(value)
        if not (0.0 <= number <= 1.0):
            raise ValueError(f"{key} must be in [0, 1].")
        return _rounded(number, round_to)

    return validate


def optional_option(inner: OptionValidator) -> OptionValidator:
    """Wrap a validator so that ``None`` passes through unchanged."""

    def validate(value: Any) -> object:
        if value is None:
            return None
        return inner(value)

    return validate


def ordered_limits_option(key: str, *, round_to: int | None = None) -> OptionValidator:
    """Validator for an ``(lower, upper)`` axis-limit pair with ``lower < upper``."""

    def validate(value: Any) -> tuple[float, float]:
        lower = float(value[0])
        upper = float(value[1])
        if not (lower < upper):
            raise ValueError(f"{key}[0] must be less than {key}[1].")
        return (_rounded(lower, round_to), _rounded(upper, round_to))

    return validate


class _ValidatedOptions:
    """Mixin providing a table-driven ``__setattr__`` for the plot options dataclasses.

    Subclasses declare ``_VALIDATORS`` mapping every field name to its validator; unknown
    attributes raise ``AttributeError``, so typos fail fast on slotted dataclasses too.
    """

    __slots__ = ()

    _VALIDATORS: ClassVar[dict[str, OptionValidator]] = {}

    def __setattr__(self, key: str, value: Any) -> None:
        validator = self._VALIDATORS.get(key)
        if validator is None:
            raise AttributeError(f"Unknown {type(self).__name__} attribute: {key}")
        object.__setattr__(self, key, validator(value))


# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _TikzColorToken:
    """Internal representation of a color token for TikZ emission.

    Attributes:
        kind (TikzColorKind): Output encoding category, as classified by
            :func:`gerrytools.latex._colors.classify_tikz_color`.
        value (str): Color payload. For ``kind="xcolor"``, this is an xcolor expression
            such as ``"denim!20!amber"``. For ``kind="html"``, this is an uppercase
            6-digit hex token such as ``"1560BD"``. For ``kind="none"``, this is ``"none"``.
    """

    kind: TikzColorKind
    value: str


# ---------------------------------------------------------------------------
# Plot base class
# ---------------------------------------------------------------------------


class _TikzPlotBase:
    """Shared document plumbing, axis setters, and TikZ command builders."""

    _options_cls: ClassVar[type]

    def __init__(self) -> None:
        self._document = TexDocument()
        self._document.add_packages("tikz")
        self.options = self._options_cls()

    def __repr__(self) -> str:  # pragma: no cover
        # `print(obj)` and the notebook repr both show the full standalone document; the body alone
        # is `document.body_string` (or `print()` on the TikZ plots).
        return self.document.to_tex()

    def __str__(self) -> str:  # pragma: no cover
        return self.document.to_tex()

    @property
    def document(self) -> TexDocument:
        """Return the LaTeX document associated with this plot.

        Returns:
            TexDocument: Document object containing the generated TikZ source.
        """
        self._document.body_string = self._generate_latex()
        return self._document

    def _generate_latex(self) -> str:
        """Render the TikZ picture body for this plot."""
        raise NotImplementedError  # pragma: no cover

    def print(self) -> None:
        """Print the raw TikZ body for this plot."""
        print(self._generate_latex())

    def preview(self) -> None:  # pragma: no cover
        """Preview the plot via its TexDocument."""
        self.document.preview()

    def clear_options(self) -> None:
        """Reset the plot options to defaults."""
        self.options = self._options_cls()

    # ==================
    #   OPTION SETTERS
    # ==================

    def set_xlim(self, xmin: float, xmax: float, rescale: bool = False) -> None:
        """Set x-axis limits.

        Args:
            xmin (float): Lower x-axis limit.
            xmax (float): Upper x-axis limit.
            rescale (bool, optional): If True, adjust xscale to preserve visual span.
                Defaults to False.
        """
        old_span = self.options.xlim[1] - self.options.xlim[0]
        new_span = float(xmax) - float(xmin)
        self.options.xlim = (xmin, xmax)
        if rescale:
            self.set_xscale(self.options.xscale * old_span / new_span)

    def set_ylim(self, ymin: float, ymax: float, rescale: bool = False) -> None:
        """Set y-axis limits.

        Args:
            ymin (float): Lower y-axis limit.
            ymax (float): Upper y-axis limit.
            rescale (bool, optional): If True, adjust yscale to preserve visual span.
                Defaults to False.
        """
        old_span = self.options.ylim[1] - self.options.ylim[0]
        new_span = float(ymax) - float(ymin)
        self.options.ylim = (ymin, ymax)
        if rescale:
            self.set_yscale(self.options.yscale * old_span / new_span)

    def set_xscale(self, xscale: float) -> None:
        """Set the TikZ xscale factor.

        Args:
            xscale (float): X-axis TikZ scale factor.
        """
        self.options.xscale = xscale

    def set_yscale(self, yscale: float) -> None:
        """Set the TikZ yscale factor.

        Args:
            yscale (float): Y-axis TikZ scale factor.
        """
        self.options.yscale = yscale

    def set_scale(self, xscale: float | None = None, yscale: float | None = None) -> None:
        """Set TikZ xscale/yscale factors.

        Args:
            xscale (float | None, optional): X-axis scale factor. Defaults to None.
            yscale (float | None, optional): Y-axis scale factor. Defaults to None.
        """
        if xscale is not None:
            self.set_xscale(xscale)
        if yscale is not None:
            self.set_yscale(yscale)

    # =====================
    #   COLOR MACHINERY
    # =====================

    def _to_latex_color(self, color: Color | None) -> _TikzColorToken:
        """Convert a color value into an internal TikZ color token.

        No color name is registered on the document: HTML hex tokens are emitted inline, either
        via a ``\\color[HTML]{...}`` scope or an extended xcolor specification in option values.

        Args:
            color (Color | None): Input color value. None is transparent.

        Returns:
            _TikzColorToken: Classified color token.
        """
        color_kind, color_value = classify_tikz_color(color)
        return _TikzColorToken(kind=color_kind, value=color_value)

    @staticmethod
    def _color_prefix(color: _TikzColorToken) -> str:
        """Build a color prefix command for a TikZ command scope.

        Args:
            color (_TikzColorToken): Internal color token.

        Returns:
            str: Color-setting prefix command or empty string for ``none``.
        """
        if color.kind == "html":
            return rf"\color[HTML]{{{color.value}}}"
        if color.kind == "xcolor":
            return rf"\color{{{color.value}}}"
        return ""

    @staticmethod
    def _wrap_with_color_scope(command: str, color: _TikzColorToken) -> str:
        """Wrap a TikZ command in a local color scope when needed.

        Args:
            command (str): TikZ command ending in ``;``.
            color (_TikzColorToken): Internal color token.

        Returns:
            str: Scoped TikZ command with color prefix, or ``command`` unchanged.
        """
        color_prefix = _TikzPlotBase._color_prefix(color)
        if len(color_prefix) == 0:
            return command
        return "{" + color_prefix + command + "}"

    @staticmethod
    def _inline_color_value(color: _TikzColorToken) -> str:
        """Render a color token as an inline value for TikZ options like ``fill=...``.

        HTML hex tokens use xcolor's extended specification, so no document-level
        ``\\definecolor`` is needed.

        Args:
            color (_TikzColorToken): Internal color token.

        Returns:
            str: Inline color expression (or ``"none"``).
        """
        if color.kind == "html":
            red, green, blue = hex_to_rgb(color.value)
            return f"{{rgb,255:red,{red};green,{green};blue,{blue}}}"
        return color.value

    # =====================
    #   COMMAND BUILDERS
    # =====================

    def _draw_path_command(
        self,
        *,
        path: str,
        color: _TikzColorToken,
        linewidth: float,
        linestyle: str | None = None,
        fill: _TikzColorToken | None = None,
        fill_opacity: float | None = None,
        draw_opacity: float | None = None,
    ) -> str:
        """Build a ``\\draw`` command with the provided styling and color token.

        Args:
            path (str): TikZ path expression without trailing semicolon.
            color (_TikzColorToken): Internal color token.
            linewidth (float): Line width in points.
            linestyle (str | None, optional): TikZ line-style token. Defaults to None.
            fill (_TikzColorToken | None): Optional fill color.
            fill_opacity (float | None): Optional fill opacity.
            draw_opacity (float | None): Optional stroke opacity.

        Returns:
            str: Fully formed TikZ ``\\draw`` command.
        """
        options = [f"line width={linewidth:0.2f}pt"]
        if linestyle is not None:
            options.append(linestyle)
        if fill is not None:
            options.append(f"fill={self._inline_color_value(fill)}")
        if fill_opacity is not None:
            options.append(f"fill opacity={fill_opacity:0.4f}")
        if draw_opacity is not None:
            options.append(f"draw opacity={draw_opacity:0.4f}")
        if color.kind == "none":
            options.append("draw=none")

        command = rf"\draw [{', '.join(options)}] {path};"
        return self._wrap_with_color_scope(command, color)

    def _fill_rectangle_command(
        self,
        *,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        color: _TikzColorToken,
        fill_opacity: float,
    ) -> str:
        """Build a ``\\fill`` rectangle command with color and opacity.

        Args:
            xmin (float): Left x-coordinate.
            ymin (float): Bottom y-coordinate.
            xmax (float): Right x-coordinate.
            ymax (float): Top y-coordinate.
            color (_TikzColorToken): Internal color token.
            fill_opacity (float): Fill opacity in ``[0, 1]``.

        Returns:
            str: Fully formed TikZ ``\\fill`` command.
        """
        options = [f"fill opacity={fill_opacity:0.4f}"]
        if color.kind == "none":
            options.append("fill=none")

        command = (
            rf"\fill [{', '.join(options)}] ({xmin:0.4f}, {ymin:0.4f}) rectangle "
            rf"({xmax:0.4f}, {ymax:0.4f});"
        )
        return self._wrap_with_color_scope(command, color)

    def _marker_node_command(
        self,
        *,
        x: float | str,
        y: float | str,
        color: _TikzColorToken,
        size_pt: float,
        edge_color: _TikzColorToken | None = None,
        fill_opacity: float | None = None,
        edge_width: float | None = None,
        edge_opacity: float | None = None,
        transform_shape: bool | None = None,
    ) -> str:
        """Build a circular marker node command.

        Args:
            x (float | str): Marker x-coordinate or TikZ coordinate expression.
            y (float | str): Marker y-coordinate or TikZ coordinate expression.
            color (_TikzColorToken): Internal color token.
            size_pt (float): Marker diameter in points.
            edge_color (_TikzColorToken | None): Optional distinct edge color.
            fill_opacity (float | None): Optional fill opacity.
            edge_width (float | None): Optional edge width.
            edge_opacity (float | None): Optional edge opacity.
            transform_shape (bool | None): Optional TikZ transform-shape setting.

        Returns:
            str: Fully formed TikZ ``\\node`` command.
        """
        x_value = x if isinstance(x, str) else f"{x:0.4f}"
        y_value = y if isinstance(y, str) else f"{y:0.4f}"
        options = ["circle", "inner sep=0pt", f"minimum size={size_pt:0.2f}pt"]
        if edge_color is None:
            if color.kind == "none":
                options.extend(["fill=none", "draw=none"])
            else:
                options.extend(["fill", "draw"])
            command = rf"\node [{', '.join(options)}] at ({x_value}, {y_value}) {{}};"
            return self._wrap_with_color_scope(command, color)

        if transform_shape is not None:
            options.insert(0, f"transform shape={str(transform_shape).lower()}")
        options.extend(
            [
                f"fill={self._inline_color_value(color)}",
                f"draw={self._inline_color_value(edge_color)}",
            ]
        )
        if fill_opacity is not None:
            options.append(f"fill opacity={fill_opacity:0.4f}")
        if edge_width is not None:
            options.append(f"line width={edge_width:0.2f}pt")
        if edge_opacity is not None:
            options.append(f"draw opacity={edge_opacity:0.4f}")
        return rf"\node [{', '.join(options)}] at ({x_value}, {y_value}) {{}};"
