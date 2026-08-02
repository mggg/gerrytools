from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, TypeAlias

from gerrytools.typing import UNSET, Color, MplKwargs, Unset

LegendAnchor: TypeAlias = tuple[float, float] | tuple[float, float, float, float]
"""Two- or four-coordinate anchor accepted by Matplotlib legends."""


@dataclass
class LegendOptions:
    """Restricted subset of Matplotlib legend options.

    Defaults place a framed, single-column legend immediately to the right of the axes.

    Attributes:
        loc (str | int): Legend location accepted by Matplotlib. Defaults to ``"center left"``.
        bbox_to_anchor (LegendAnchor | None): Anchor coordinates, or None for no explicit anchor.
            Defaults to ``(1.01, 0.5)``.
        ncols (int): Number of legend columns. Defaults to 1.
        fontsize (float | str | None): Label font size, or None for Matplotlib's default. Defaults
            to None.
        frameon (bool): Whether to draw the legend frame. Defaults to True.
        fancybox (bool): Whether to round the frame corners. Defaults to False.
        shadow (bool): Whether to draw a frame shadow. Defaults to False.
        framealpha (float | None): Optional frame-opacity override. Defaults to None.
        facecolor (Color | None | Unset): Frame fill. An unset value uses Matplotlib's default;
            None removes the fill. Defaults to unset.
        edgecolor (Color | None | Unset): Frame edge. An unset value uses Matplotlib's default;
            None removes the edge. Defaults to unset.
        title (str | None): Legend title, or None for no title. Defaults to None.
        alignment (Literal["center", "left", "right"]): Legend content alignment. Defaults to
            ``"center"``.
        labelspacing (float): Vertical spacing between entries in font-size units. Defaults to 0.5.
        columnspacing (float): Horizontal spacing between columns in font-size units. Defaults to
            2.0.
    """

    loc: str | int = "center left"
    bbox_to_anchor: LegendAnchor | None = (1.01, 0.5)
    ncols: int = 1
    fontsize: float | str | None = None
    frameon: bool = True
    fancybox: bool = False
    shadow: bool = False
    framealpha: float | None = None
    facecolor: Color | None | Unset = UNSET
    edgecolor: Color | None | Unset = UNSET
    title: str | None = None
    alignment: Literal["center", "left", "right"] = "center"
    labelspacing: float = 0.5
    columnspacing: float = 2.0

    def to_dict(self) -> MplKwargs:
        """Convert to keyword arguments accepted by Matplotlib's ``Axes.legend``.

        Returns:
            MplKwargs: Legend keyword arguments with unset and None values omitted.
        """
        output: MplKwargs = {}
        for field in fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, Unset):
                continue
            if field.name in {"facecolor", "edgecolor"} and field_value is None:
                output[field.name] = "none"
            elif field_value is not None:
                output[field.name] = field_value
        return output
