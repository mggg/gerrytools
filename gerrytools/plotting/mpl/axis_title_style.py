from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, cast

from gerrytools.colors import resolve_color_and_alpha, resolve_rgba
from gerrytools.logging import get_logger
from gerrytools.plotting.utils import _validated_finite, _validated_nonneg_finite
from gerrytools.typing import Color, MplKwargs, MplRGBAColor

logger = get_logger(__name__)


class AxisLabelKwargs(TypedDict, total=False):
    """Keyword arguments for ``Axes.set_xlabel`` and ``Axes.set_ylabel``."""

    color: MplRGBAColor
    fontsize: float | int
    fontweight: str
    fontstyle: Literal["normal", "italic", "oblique"]
    fontfamily: str
    labelpad: float


class TitleKwargs(TypedDict, total=False):
    """Keyword arguments for ``Axes.set_title``."""

    color: MplRGBAColor
    fontsize: float | int
    fontweight: str
    fontstyle: Literal["normal", "italic", "oblique"]
    fontfamily: str
    loc: Literal["left", "center", "right"]
    pad: float


@dataclass(frozen=True)
class _FontStyleBase:
    """Shared font fields and validation for the label/title style dataclasses."""

    fontsize: float | int | None = None
    fontweight: str | None = None
    fontstyle: Literal["normal", "italic", "oblique"] | None = None
    fontfamily: str | None = None

    fontcolor: Color | None = "black"
    fontalpha: float | None = None

    def __post_init__(self) -> None:
        if self.fontsize is not None:
            object.__setattr__(
                self,
                "fontsize",
                _validated_nonneg_finite(self.fontsize, field=f"{type(self).__name__}.fontsize"),
            )

        resolved_color, resolved_alpha = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=True,
            field="fontcolor",
            owner=type(self).__name__,
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_color)
        object.__setattr__(self, "fontalpha", resolved_alpha)

    def _font_settings(self) -> MplKwargs:
        """The shared font kwargs, with unset (None) fields dropped."""
        settings: MplKwargs = {"color": resolve_rgba(self.fontcolor, self.fontalpha)}
        for name in ("fontsize", "fontweight", "fontstyle", "fontfamily"):
            value = getattr(self, name)
            if value is not None:
                settings[name] = value
        return settings


@dataclass(frozen=True)
class AxisLabelStyle(_FontStyleBase):
    """Matplotlib style options for axis labels.

    Attributes:
        fontsize (float | int | None): Font size, or None to use Matplotlib's default. Defaults to
            None.
        fontweight (str | None): Font weight, or None to use Matplotlib's default. Defaults to
            None.
        fontstyle (Literal["normal", "italic", "oblique"] | None): Font slant, or None to use
            Matplotlib's default. Defaults to None.
        fontfamily (str | None): Font family, or None to use Matplotlib's default. Defaults to
            None.
        fontcolor (Color | None): Label color. None removes the color. Defaults to ``"black"``.
        fontalpha (float | None): Optional font-opacity override. Defaults to None.
        labelpad (float | None): Distance from the axis in points. Negative values move the label
            inward. Defaults to None.

    Raises:
        ValueError: If a font value, pad, color, or alpha is invalid.
    """

    labelpad: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.labelpad is not None:
            # Negative pads are legal in matplotlib (they pull the label inward).
            object.__setattr__(
                self,
                "labelpad",
                _validated_finite(self.labelpad, field="AxisLabelStyle.labelpad"),
            )

    def to_mpl_settings_dict(self) -> AxisLabelKwargs:
        """Convert to Matplotlib keyword arguments for setting an axis label.

        Returns:
            AxisLabelKwargs: Resolved axis-label keyword arguments.
        """
        settings_dict = cast("AxisLabelKwargs", self._font_settings())
        if self.labelpad is not None:
            settings_dict["labelpad"] = self.labelpad
        return settings_dict


@dataclass(frozen=True)
class TitleStyle(_FontStyleBase):
    """Matplotlib style options for axes titles.

    Attributes:
        fontsize (float | int | None): Font size, or None to use Matplotlib's default. Defaults to
            None.
        fontweight (str | None): Font weight, or None to use Matplotlib's default. Defaults to
            None.
        fontstyle (Literal["normal", "italic", "oblique"] | None): Font slant, or None to use
            Matplotlib's default. Defaults to None.
        fontfamily (str | None): Font family, or None to use Matplotlib's default. Defaults to
            None.
        fontcolor (Color | None): Title color. None removes the color. Defaults to ``"black"``.
        fontalpha (float | None): Optional font-opacity override. Defaults to None.
        loc (Literal["left", "center", "right"] | None): Title alignment, or None to use
            Matplotlib's default. Defaults to None.
        pad (float | None): Distance above the axes in points. Negative values move the title
            inward. Defaults to None.

    Raises:
        ValueError: If a font value, pad, color, alpha, or title location is invalid.
    """

    loc: Literal["left", "center", "right"] | None = None
    pad: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pad is not None:
            # Negative pads are legal in matplotlib (they pull the title inward).
            object.__setattr__(self, "pad", _validated_finite(self.pad, field="TitleStyle.pad"))
        if self.loc is not None and self.loc not in ("left", "center", "right"):
            raise ValueError("TitleStyle.loc must be one of {'left','center','right'}.")

    def to_mpl_settings_dict(self) -> TitleKwargs:
        """Convert to Matplotlib keyword arguments for ``Axes.set_title``.

        Returns:
            TitleKwargs: Resolved title keyword arguments.
        """
        settings_dict = cast("TitleKwargs", self._font_settings())
        if self.loc is not None:
            settings_dict["loc"] = self.loc
        if self.pad is not None:
            settings_dict["pad"] = self.pad
        return settings_dict
