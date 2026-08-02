from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Sequence, Union

from matplotlib.colors import to_hex

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.plotting.utils import _validated_nonneg_finite
from gerrytools.typing import Color

FontStyle = Literal["normal", "italic", "oblique"]
"""How glyphs are slanted."""

FontVariant = Literal["normal", "small-caps"]
"""Glyph variant selection."""

FontStretchName = Literal[
    "ultra-condensed",
    "extra-condensed",
    "condensed",
    "semi-condensed",
    "normal",
    "semi-expanded",
    "expanded",
    "extra-expanded",
    "ultra-expanded",
]
FontStretch = Union[FontStretchName, int, float]
"""Width of the font face (condensed/expanded)."""

FontWeightName = Literal[
    "ultralight",
    "light",
    "normal",
    "regular",
    "book",
    "medium",
    "roman",
    "semibold",
    "demibold",
    "demi",
    "bold",
    "heavy",
    "extra bold",
    "black",
]
FontWeight = Union[FontWeightName, int, float]
"""Stroke thickness / darkness of glyphs."""

GenericFontFamily = Literal[
    "serif",
    "sans-serif",
    "sans serif",
    "sans",
    "monospace",
    "cursive",
    "fantasy",
]
FontFamily = Union[GenericFontFamily, str, Sequence[str]]
"""Font family selection."""


@dataclass(frozen=True, slots=True)
class LabelFontOptions:
    """Font options for text labels.

    Attributes:
        fontcolor (Color | None): Text color. None removes the color. Defaults to ``"white"``.
        fontalpha (float | None): Optional text-opacity override. Defaults to 1.0.
        fontsize (float): Font size in points. Defaults to 6.0.
        fontweight (FontWeight): Font weight. Defaults to ``"bold"``.
        fontstyle (FontStyle): Font slant. Defaults to ``"normal"``.
        fontvariant (FontVariant): Font variant. Defaults to ``"normal"``.
        fontstretch (FontStretch | None): Font width, or None for the Matplotlib default. Defaults
            to None.
        fontfamily (FontFamily | None): Font family, or None for the Matplotlib default. Defaults
            to None.
        outlinecolor (Color | None): Text-outline color. None removes the outline. Defaults to
            ``"black"``.
        outlinewidth (float): Text-outline width in points. Defaults to 0.75.

    Raises:
        ValueError: If a font, color, alpha, or outline-width value is invalid.
    """

    fontcolor: Color | None = "white"
    fontalpha: float | None = 1.0
    fontsize: float = 6.0

    fontweight: FontWeight = "bold"
    fontstyle: FontStyle = "normal"
    fontvariant: FontVariant = "normal"
    fontstretch: FontStretch | None = None
    fontfamily: FontFamily | None = None

    outlinecolor: Color | None = "black"
    outlinewidth: float = 0.75

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fontsize",
            _validated_nonneg_finite(self.fontsize, field="LabelFontOptions.fontsize"),
        )
        object.__setattr__(
            self,
            "outlinewidth",
            _validated_nonneg_finite(self.outlinewidth, field="LabelFontOptions.outlinewidth"),
        )

    def to_mpl_text_kwargs(self) -> dict:
        """Return keyword arguments for ``Axes.text`` font styling.

        Returns:
            dict: Matplotlib text keyword arguments.
        """
        font_color, font_alpha = resolve_color_and_alpha(self.fontcolor, self.fontalpha)
        kw: dict = {
            "color": to_hex((font_color, font_alpha), keep_alpha=True),
            "fontsize": float(self.fontsize),
            "fontweight": self.fontweight,
            "fontstyle": self.fontstyle,
            "fontvariant": self.fontvariant,
        }
        if self.fontstretch is not None:
            kw["fontstretch"] = self.fontstretch
        if self.fontfamily is not None:
            kw["fontfamily"] = self.fontfamily
        return kw


@dataclass(frozen=True, slots=True)
class LabelBoxOptions:
    """Background box options for text labels drawn via ``Axes.text(..., bbox=...)``.

    Attributes:
        enabled (bool): Whether to draw the box. Defaults to True.
        boxstyle (Literal["square", "round", "round4", "circle", "ellipse"]): Matplotlib box
            shape. Defaults to ``"round4"``.
        pad (float): Padding around the text. Defaults to 0.25.
        facecolor (Color | None): Fill color. None removes the fill. Defaults to ``"black"``.
        facealpha (float | None): Optional fill-opacity override. Defaults to 0.6.
        edgecolor (Color | None): Border color. None removes the border. Defaults to ``"none"``.
        edgealpha (float | None): Optional border-opacity override. Defaults to 0.0.
        edgewidth (float): Border width in points. Defaults to 0.8.
    """

    enabled: bool = True
    boxstyle: Literal["square", "round", "round4", "circle", "ellipse"] = "round4"
    pad: float = 0.25
    facecolor: Color | None = "black"
    facealpha: float | None = 0.6
    edgecolor: Color | None = "none"
    edgealpha: float | None = 0.0
    edgewidth: float = 0.8

    def to_mpl_bbox(self) -> dict | None:
        """Return a dictionary suitable for passing as ``bbox`` to ``Axes.text``.

        Returns:
            dict | None: Matplotlib bounding-box arguments, or None when the box is disabled.
        """
        if not self.enabled:
            return None

        face_color, face_alpha = resolve_color_and_alpha(self.facecolor, alpha=self.facealpha)
        edge_color, edge_alpha = resolve_color_and_alpha(self.edgecolor, alpha=self.edgealpha)

        return {
            "boxstyle": f"{self.boxstyle},pad={float(self.pad)}",
            "fc": to_hex((face_color, face_alpha), keep_alpha=True),
            "ec": to_hex((edge_color, edge_alpha), keep_alpha=True),
            "lw": float(self.edgewidth),
        }


@dataclass(frozen=True, slots=True)
class LabelStyle:
    """A named bundle of label font and box styling for map labels.

    Attributes:
        font (LabelFontOptions): Font options applied to every label. Defaults to
            ``LabelFontOptions()``.
        box (LabelBoxOptions | None): Background box options, or None for no box.
        equalize_circle_pad (bool): When True and ``box`` uses the "circle" boxstyle, shorter labels
            get extra pad so one- and two-character labels render as circles of the same diameter.
    """

    font: LabelFontOptions = field(default_factory=LabelFontOptions)
    box: LabelBoxOptions | None = None
    equalize_circle_pad: bool = False

    def with_font(self, **overrides: object) -> "LabelStyle":
        """A copy of this style with the given font fields replaced.

        Args:
            **overrides (object): ``LabelFontOptions`` field values, e.g. ``fontsize=12``.

        Returns:
            LabelStyle: The tweaked style; the box and pad behavior are unchanged.
        """
        return replace(self, font=replace(self.font, **overrides))

    def with_box(self, **overrides: object) -> "LabelStyle":
        """A copy of this style with the given box fields replaced.

        Args:
            **overrides (object): ``LabelBoxOptions`` field values, e.g. ``facealpha=0.9``.

        Returns:
            LabelStyle: The tweaked style.

        Raises:
            ValueError: If this style has no box to tweak.
        """
        if self.box is None:
            raise ValueError(
                f"Label style has no box; construct a LabelStyle with a LabelBoxOptions "
                f"instead of tweaking one. Got overrides: {sorted(overrides)}."
            )
        return replace(self, box=replace(self.box, **overrides))

    def box_for(self, label: str) -> LabelBoxOptions | None:
        """The box options for one label.

        Args:
            label (str): The label text the box will enclose.

        Returns:
            LabelBoxOptions | None: The style's box options, padded up for short labels when
                ``equalize_circle_pad`` is set, or None when the style has no box.
        """
        if self.box is None or not self.equalize_circle_pad:
            return self.box
        extra_pad = 0.25 if len(str(label)) < 2 else 0.0
        return replace(self.box, pad=self.box.pad + extra_pad)


LABEL_STYLES: dict[str, LabelStyle] = {
    # White bold text with a black outline, the districtr-style plan-map look. Larger than the raw
    # defaults so the halo reads at map scale.
    "halo": LabelStyle(
        font=LabelFontOptions(fontsize=9, outlinewidth=1.5),
    ),
    # District-number badges: bold black text in a wheat circle with a black rim, the convention in
    # recent expert-report maps.
    "badge": LabelStyle(
        font=LabelFontOptions(
            fontsize=8,
            fontweight="bold",
            fontcolor="black",
            outlinewidth=0,
        ),
        box=LabelBoxOptions(
            boxstyle="circle",
            pad=0.3,
            facecolor="#f5deb3",
            facealpha=1.0,
            edgecolor="black",
            edgealpha=1.0,
            edgewidth=0.5,
        ),
        equalize_circle_pad=True,
    ),
    # Bold black text with a white outline: the inverse of "halo", for names over light fills such
    # as county labels on a pale choropleth.
    "ink": LabelStyle(
        font=LabelFontOptions(
            fontsize=8,
            fontweight="bold",
            fontcolor="black",
            outlinecolor="white",
            outlinewidth=2,
        ),
    ),
    # Small unadorned black text, the quiet place-name style on plan maps whose fills carry the
    # information.
    "plain": LabelStyle(
        font=LabelFontOptions(
            fontsize=8,
            fontweight="roman",
            fontcolor="black",
            outlinewidth=0,
        ),
    ),
    # Bold black text in a translucent white rounded box with a black rim: a name tag that stays
    # readable over any fill without fully hiding the geography beneath it.
    "tag": LabelStyle(
        font=LabelFontOptions(
            fontsize=8,
            fontweight="bold",
            fontcolor="black",
            outlinewidth=0,
        ),
        box=LabelBoxOptions(
            boxstyle="round4",
            pad=0.35,
            facecolor="white",
            facealpha=0.75,
            edgecolor="black",
            edgealpha=1.0,
            edgewidth=0.5,
        ),
    ),
}
"""Named label styles accepted wherever a ``label_style=`` argument takes a string."""


def resolve_label_style(label_style: LabelStyle | str) -> LabelStyle:
    """Resolve a style name or ``LabelStyle`` instance to a ``LabelStyle``.

    Args:
        label_style (LabelStyle | str): A ``LabelStyle`` or the name of a registered style.

    Returns:
        LabelStyle: The resolved style.

    Raises:
        ValueError: If ``label_style`` is a string that names no registered style.
    """
    if isinstance(label_style, LabelStyle):
        return label_style
    try:
        return LABEL_STYLES[label_style]
    except KeyError:
        raise ValueError(
            f"Unknown label style {label_style!r}; available styles: {sorted(LABEL_STYLES)}."
        ) from None
