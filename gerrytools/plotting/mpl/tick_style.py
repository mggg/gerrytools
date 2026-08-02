from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.utils import _validated_nonneg_finite
from gerrytools.typing import Color, TickType

logger = get_logger(__name__)


@dataclass(frozen=True)
class TickStyle:
    """Style options for axis ticks and their labels.

    Attributes:
        size (float | int): Tick-label font size. Defaults to 10.
        rotation (float | int): Tick-label rotation in degrees. Defaults to 0.
        fontcolor (Color | None): Tick-label color. None removes the color. Defaults to
            ``"black"``.
        fontalpha (float | None): Optional tick-label opacity override. Defaults to None.
        tickcolor (Color | None): Tick-mark color. None removes the color. Defaults to
            ``"black"``.
        tickalpha (float | None): Optional tick-mark opacity override. Defaults to None.
        fontweight (str): Tick-label font weight. Defaults to ``"normal"``.
        fontstyle (Literal["normal", "italic", "oblique"]): Tick-label slant. Defaults to
            ``"normal"``.
        fontfamily (str): Tick-label font family. Defaults to ``"sans-serif"``.
        ticktype (TickType): Whether to style major, minor, or both ticks. Defaults to ``"major"``.

    Raises:
        ValueError: If a size, color, alpha, or tick type is invalid.
    """

    size: float | int = 10
    rotation: float | int = 0
    fontcolor: Color | None = "black"
    fontalpha: float | None = None
    tickcolor: Color | None = "black"
    tickalpha: float | None = None
    fontweight: str = "normal"
    fontstyle: Literal["normal", "italic", "oblique"] = "normal"
    fontfamily: str = "sans-serif"
    ticktype: TickType = "major"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "size", _validated_nonneg_finite(self.size, field="TickStyle.size")
        )

        resolved_fc, resolved_fa = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=True,
            field="fontcolor",
            owner="TickStyle",
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_fc)
        object.__setattr__(self, "fontalpha", resolved_fa)

        resolved_tc, resolved_ta = resolve_color_and_alpha(
            self.tickcolor,
            self.tickalpha,
            allow_none=True,
            field="tickcolor",
            owner="TickStyle",
            logger=logger,
        )
        object.__setattr__(self, "tickcolor", resolved_tc)
        object.__setattr__(self, "tickalpha", resolved_ta)

        if self.ticktype not in ("major", "minor", "both"):
            raise ValueError("TickStyle.ticktype must be 'major', 'minor', or 'both'.")
