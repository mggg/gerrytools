import math
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting.data.options import BandOptions, LineOptions
from gerrytools.plotting.mpl.label_text_options import (
    FontFamily,
    FontStyle,
    FontWeight,
    LabelBoxOptions,
    LabelFontOptions,
)
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.plotting.utils import (
    _resolve_color_clamped_width,
    _validated_finite,
    _validated_nonneg_finite,
)
from gerrytools.typing import Color

logger = get_logger(__name__)


def _set_finite(instance: object, name: str, *, owner: str) -> None:
    """Validate a frozen dataclass field as finite and write the float back."""
    value = _validated_finite(getattr(instance, name), field=f"{owner}.{name}")
    object.__setattr__(instance, name, value)


def _set_nonneg_finite(instance: object, name: str, *, owner: str) -> None:
    """Validate a frozen dataclass field as nonnegative and finite."""
    value = _validated_nonneg_finite(getattr(instance, name), field=f"{owner}.{name}")
    object.__setattr__(instance, name, value)


@dataclass(frozen=True)
class _PointSetData:
    """A dataclass representing a set of points to be plotted on a boxplot figure.

    Attributes:
        name (str): The name of the point set.
        values_dict (dict[str, float]): A dictionary mapping labels to point values.
        point_data (PointMarkerOptions): The settings for the points.
        x_offset (float | None): An optional absolute x-offset from category center.
    """

    name: str
    values_dict: dict[str, float]  # one value per label
    point_data: PointMarkerOptions
    x_offset: float | None = None  # optional absolute x-offset from category center


@dataclass(frozen=True)
class _LineData:
    """One vertical/horizontal line annotation: positions plus resolved styling.

    Attributes:
        values (tuple[float, ...]): The position(s) of the line(s) on the axis.
        style (LineOptions): Resolved line styling.
        name (str | None): The name of the line for legend purposes.
    """

    values: tuple[float, ...]
    style: LineOptions
    name: str | None = None


@dataclass(frozen=True)
class _BandData:
    """One vertical/horizontal band annotation: bounds plus resolved styling.

    Attributes:
        lower_bound (float): The lower bound of the band.
        upper_bound (float): The upper bound of the band.
        style (BandOptions): Resolved band styling.
        name (str | None): The name of the band for legend purposes.
    """

    lower_bound: float
    upper_bound: float
    style: BandOptions
    name: str | None = None

    def __post_init__(self) -> None:
        lower, upper = sorted([float(self.lower_bound), float(self.upper_bound)])
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError("_BandData: lower_bound and upper_bound must both be finite.")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)


@dataclass(frozen=True)
class ArrowTextStyle:
    """Text styling options for annotation arrows.

    Attributes:
        fontsize (float, optional): Text size in points. Defaults to ``10.0``.
        fontcolor (Color | None, optional): Text color. None makes the text transparent.
            Defaults to ``"black"``.
        fontalpha (float | None, optional): Optional alpha override for ``fontcolor``.
            Defaults to None.
        fontoutlinecolor (Color | None, optional): Outline color for text glyphs. None removes the
            outline. Defaults to ``"black"``.
        fontoutlinealpha (float | None, optional): Optional alpha override for
            ``fontoutlinecolor``. Defaults to None.
        fontoutlinewidth (float, optional): Outline width in points. Defaults to ``0.5``.
        fontweight (FontWeight | None, optional): Font weight (for example ``"bold"``).
            Defaults to None.
        fontstyle (FontStyle | None, optional): Font style.
            Defaults to None.
        fontfamily (FontFamily | None, optional): Font family selection.
            Defaults to None.
        rotation (float | None, optional): Text rotation in degrees. Defaults to None.
        horizontalalignment (Literal["left", "center", "right"] | None, optional):
            Horizontal text alignment. Defaults to None.
        verticalalignment (Literal["bottom", "center", "top"] | None, optional):
            Vertical text alignment. Defaults to None.

    Raises:
        ValueError: If a font, color, alpha, outline, rotation, or alignment value is invalid.
    """

    fontsize: float = 10.0
    fontcolor: Color | None = "black"
    fontalpha: float | None = None
    fontoutlinecolor: Color | None = "black"
    fontoutlinealpha: float | None = None
    fontoutlinewidth: float = 0.5
    fontweight: FontWeight | None = None
    fontstyle: FontStyle | None = None
    fontfamily: FontFamily | None = None
    rotation: float | None = None
    horizontalalignment: Literal["left", "center", "right"] | None = None
    verticalalignment: Literal["bottom", "center", "top"] | None = None

    def __post_init__(self) -> None:
        _set_nonneg_finite(self, "fontsize", owner="ArrowTextStyle")
        _set_nonneg_finite(self, "fontoutlinewidth", owner="ArrowTextStyle")
        if self.rotation is not None:
            _set_finite(self, "rotation", owner="ArrowTextStyle")

        resolved_fc, resolved_fa = resolve_color_and_alpha(
            self.fontcolor,
            self.fontalpha,
            allow_none=True,
            field="fontcolor",
            owner="ArrowTextStyle",
            logger=logger,
        )
        object.__setattr__(self, "fontcolor", resolved_fc)
        object.__setattr__(self, "fontalpha", resolved_fa)

        if self.fontoutlinecolor is None:
            object.__setattr__(self, "fontoutlinealpha", None)
            if self.fontoutlinewidth > 0:
                logger.debug(
                    "ArrowTextStyle: fontoutlinewidth is %s but fontoutlinecolor is None; setting fontoutlinewidth to 0.",
                    self.fontoutlinewidth,
                )
                object.__setattr__(self, "fontoutlinewidth", 0.0)
            return

        resolved_oc, resolved_oa, outlinewidth = _resolve_color_clamped_width(
            self.fontoutlinecolor,
            self.fontoutlinealpha,
            self.fontoutlinewidth,
            color_field="fontoutlinecolor",
            width_field="fontoutlinewidth",
            owner="ArrowTextStyle",
            log=logger,
        )
        object.__setattr__(self, "fontoutlinecolor", resolved_oc)
        object.__setattr__(self, "fontoutlinealpha", resolved_oa)
        object.__setattr__(self, "fontoutlinewidth", outlinewidth)


@dataclass(frozen=True)
class TextArrowStyle:
    """Styling options for text arrows rendered via ``Axes.text(..., bbox=...)``.

    Attributes:
        arrowfacecolor (Color | None, optional): Fill color of the arrow box. None removes
            the fill. Defaults to ``"#5c676f"``.
        arrowfacealpha (float | None, optional): Optional alpha override for ``arrowfacecolor``.
            Defaults to None.
        arrowedgecolor (Color | None, optional): Edge color of the arrow box. None removes
            the edge. Defaults to ``"black"``.
        arrowedgealpha (float | None, optional): Optional alpha override for
            ``arrowedgecolor``. Defaults to None.
        arrowedgewidth (float, optional): Edge width in points. Defaults to ``1.0``.
        boxpad (float, optional): Text-box pad used by Matplotlib ``boxstyle``.
            Defaults to ``0.3``.
        boxstyle (str | None, optional): Explicit boxstyle override.
            When None, GerryPlot selects a direction-aware default. Defaults to None.

    Raises:
        ValueError: If a color, alpha, width, or padding value is invalid.
    """

    arrowfacecolor: Color | None = "#5c676f"
    arrowfacealpha: float | None = None
    arrowedgecolor: Color | None = "black"
    arrowedgealpha: float | None = None
    arrowedgewidth: float = 1.0
    boxpad: float = 0.3
    boxstyle: str | None = None

    def __post_init__(self) -> None:
        _set_nonneg_finite(self, "arrowedgewidth", owner="TextArrowStyle")
        _set_nonneg_finite(self, "boxpad", owner="TextArrowStyle")

        resolved_fc, resolved_fa = resolve_color_and_alpha(
            self.arrowfacecolor,
            self.arrowfacealpha,
            allow_none=True,
            field="arrowfacecolor",
            owner="TextArrowStyle",
            logger=logger,
        )
        object.__setattr__(self, "arrowfacecolor", resolved_fc)
        object.__setattr__(self, "arrowfacealpha", resolved_fa)

        resolved_ec, resolved_ea, outlinewidth = _resolve_color_clamped_width(
            self.arrowedgecolor,
            self.arrowedgealpha,
            self.arrowedgewidth,
            color_field="arrowedgecolor",
            width_field="arrowedgewidth",
            owner="TextArrowStyle",
            log=logger,
        )
        object.__setattr__(self, "arrowedgecolor", resolved_ec)
        object.__setattr__(self, "arrowedgealpha", resolved_ea)
        object.__setattr__(self, "arrowedgewidth", outlinewidth)


@dataclass(frozen=True)
class LabelArrowStyle:
    """Styling options for label arrows rendered via ``Axes.annotate``.

    Attributes:
        arrowstyle (str, optional): Matplotlib arrowstyle string.
            Defaults to ``"-|>"``.
        connectionstyle (str | None, optional): Matplotlib connectionstyle string.
            Defaults to ``"arc3"``.
        arrowhead_scale (float, optional): Arrow-head scale. Defaults to ``12.0``.
        shrink_a (float, optional): Shrink amount at the text/tail end in points.
            Defaults to ``0.0``.
        shrink_b (float, optional): Shrink amount at the tip end in points.
            Defaults to ``0.0``.
        arrowfacecolor (Color | None, optional): Face color of the arrow head. None removes
            the fill.
            Defaults to ``"#5c676f"``.
        arrowfacealpha (float | None, optional): Optional alpha override for ``arrowfacecolor``.
            Defaults to None.
        arrowedgecolor (Color | None, optional): Outline color of the arrow. None removes
            the edge.
            Defaults to ``"black"``.
        arrowedgealpha (float | None, optional): Optional alpha override for
            ``arrowedgecolor``. Defaults to None.
        arrowedgewidth (float, optional): Arrow outline width in points.
            Defaults to ``1.0``.
        linestyle (str, optional): Arrow line style. Defaults to ``"-"``.

    Raises:
        ValueError: If a color, alpha, width, shrink, or arrow-scale value is invalid.
    """

    arrowstyle: str = "-|>"
    connectionstyle: str | None = "arc3"
    arrowhead_scale: float = 12.0
    shrink_a: float = 0.0
    shrink_b: float = 0.0
    arrowfacecolor: Color | None = "#5c676f"
    arrowfacealpha: float | None = None
    arrowedgecolor: Color | None = "black"
    arrowedgealpha: float | None = None
    arrowedgewidth: float = 1.0
    linestyle: str = "-"

    def __post_init__(self) -> None:
        for name in ("arrowhead_scale", "shrink_a", "shrink_b", "arrowedgewidth"):
            _set_nonneg_finite(self, name, owner="LabelArrowStyle")

        resolved_fc, resolved_fa = resolve_color_and_alpha(
            self.arrowfacecolor,
            self.arrowfacealpha,
            allow_none=True,
            field="arrowfacecolor",
            owner="LabelArrowStyle",
            logger=logger,
        )
        object.__setattr__(self, "arrowfacecolor", resolved_fc)
        object.__setattr__(self, "arrowfacealpha", resolved_fa)

        resolved_ec, resolved_ea, outlinewidth = _resolve_color_clamped_width(
            self.arrowedgecolor,
            self.arrowedgealpha,
            self.arrowedgewidth,
            color_field="arrowedgecolor",
            width_field="arrowedgewidth",
            owner="LabelArrowStyle",
            log=logger,
        )
        object.__setattr__(self, "arrowedgecolor", resolved_ec)
        object.__setattr__(self, "arrowedgealpha", resolved_ea)
        object.__setattr__(self, "arrowedgewidth", outlinewidth)


@dataclass(frozen=True)
class ArrowPlacement:
    """Placement options for annotation arrows.

    Attributes:
        coordinate_system (Literal["data", "axes fraction"], optional): Coordinate system used
            for ``arrowtip``, ``arrowtail`` and ``text_offset``. Defaults to ``"data"``.
        text_offset (tuple[float, float], optional): Offset applied to text anchor position in
            the selected coordinate system. Defaults to ``(0.0, 0.0)``.
        label_padding (float, optional): Additional distance between a label arrow tail and its
            auto-placed text anchor, applied away from the arrow direction. Ignored when
            ``label_position`` is set explicitly. Defaults to ``0.005``.
        arrowtail (tuple[float, float] | None, optional): Explicit tail coordinate for label
            arrows. If None, GerryPlot computes tail from direction and ``tail_length``.
            Defaults to None.
        tail_length (float, optional): Tail-to-tip distance used when ``arrowtail`` is None.
            Defaults to ``0.08``.
        zorder (int | float, optional): Draw order; coerced to int. Defaults to ``20``.
        clip_on (bool, optional): Whether artists should be clipped to the axes patch.
            Defaults to False.

    Raises:
        ValueError: If a distance, offset, or explicit arrow-tail coordinate is invalid.
    """

    coordinate_system: Literal["data", "axes fraction"] = "data"
    text_offset: tuple[float, float] = (0.0, 0.0)
    label_padding: float = 0.005
    arrowtail: tuple[float, float] | None = None
    tail_length: float = 0.08
    zorder: int | float = 20
    clip_on: bool = False

    def __post_init__(self) -> None:
        _set_nonneg_finite(self, "tail_length", owner="ArrowPlacement")
        _set_nonneg_finite(self, "label_padding", owner="ArrowPlacement")
        object.__setattr__(self, "zorder", int(self.zorder))

        text_offset = tuple(
            _validated_finite(component, field="ArrowPlacement.text_offset")
            for component in self.text_offset
        )
        object.__setattr__(self, "text_offset", text_offset)

        if self.arrowtail is not None:
            arrowtail = (float(self.arrowtail[0]), float(self.arrowtail[1]))
            if not (math.isfinite(arrowtail[0]) and math.isfinite(arrowtail[1])):
                raise ValueError("ArrowPlacement.arrowtail components must be finite.")
            object.__setattr__(self, "arrowtail", arrowtail)


@dataclass(frozen=True)
class TextArrowOptions:
    """Advanced options for a text-style annotation arrow.

    Attributes:
        placement (ArrowPlacement, optional): Placement, clipping, and draw-order options.
            Defaults to ``ArrowPlacement()``.
        style (TextArrowStyle, optional): Arrow-box fill, outline, and shape styling. Defaults to
            ``TextArrowStyle()``.
    """

    placement: ArrowPlacement = field(default_factory=ArrowPlacement)
    style: TextArrowStyle = field(default_factory=TextArrowStyle)


@dataclass(frozen=True)
class LabelArrowOptions:
    """Advanced options for a label-style annotation arrow.

    Attributes:
        arrow_length (float | None, optional): Arrow length as a percentage of the axes span in
            the arrow direction. When None, ``placement.tail_length`` is used. Defaults to None.
        placement (ArrowPlacement, optional): Tail placement, label padding, clipping, and draw
            order. Defaults to an arrow with a tail length of ``0.04``.
        style (LabelArrowStyle, optional): Arrowhead, outline, and line styling. Defaults to
            ``LabelArrowStyle()``.

    Raises:
        ValueError: If ``arrow_length`` is invalid or conflicts with an explicit arrow tail.
    """

    arrow_length: float | None = None
    placement: ArrowPlacement = field(default_factory=lambda: ArrowPlacement(tail_length=0.04))
    style: LabelArrowStyle = field(default_factory=LabelArrowStyle)

    def __post_init__(self) -> None:
        if self.arrow_length is None:
            return
        arrow_length = float(self.arrow_length)
        if not math.isfinite(arrow_length):
            raise ValueError("LabelArrowOptions.arrow_length must be finite.")
        if not 0.0 <= arrow_length <= 100.0:
            raise ValueError("LabelArrowOptions.arrow_length must be in [0, 100].")
        if self.placement.arrowtail is not None:
            raise ValueError(
                "LabelArrowOptions.arrow_length cannot be set when placement.arrowtail is set."
            )
        object.__setattr__(self, "arrow_length", arrow_length)


def _validated_point(value: tuple[float, float], *, field_name: str) -> tuple[float, float]:
    """Coerce a 2-tuple to floats and require finite components."""
    point = (float(value[0]), float(value[1]))
    if not (math.isfinite(point[0]) and math.isfinite(point[1])):
        raise ValueError(f"{field_name} components must be finite.")
    return point


def _validate_arrow_direction(direction: object) -> None:
    if not isinstance(direction, str) or direction not in ("right", "left", "up", "down"):
        raise ValueError(
            f"direction must be one of 'right', 'left', 'up', or 'down'; got {direction!r}."
        )


@dataclass(frozen=True)
class _TextArrowData:
    """A deferred text-style arrow: text rendered inside an arrow-shaped box.

    Attributes:
        arrowtip (tuple[float, float]): Arrow tip coordinate in ``placement.coordinate_system``.
        direction (Literal["right", "left", "up", "down"]): Arrow direction.
        text (str): Text shown inside the arrow box.
        textstyle (ArrowTextStyle): Text style options.
        placement (ArrowPlacement): Placement options.
        style (TextArrowStyle): Arrow box styling.
        name (str | None): Optional identifier for callers.
    """

    arrowtip: tuple[float, float]
    direction: Literal["right", "left", "up", "down"]
    text: str = "   "
    textstyle: ArrowTextStyle = field(default_factory=ArrowTextStyle)
    placement: ArrowPlacement = field(default_factory=ArrowPlacement)
    style: TextArrowStyle = field(default_factory=TextArrowStyle)
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_arrow_direction(self.direction)
        object.__setattr__(
            self, "arrowtip", _validated_point(self.arrowtip, field_name="_TextArrowData.arrowtip")
        )


@dataclass(frozen=True)
class _LabelArrowData:
    """A deferred label-style arrow: a true annotation arrow plus an optional text label.

    Attributes:
        arrowtip (tuple[float, float]): Arrow tip coordinate in ``placement.coordinate_system``.
        direction (Literal["right", "left", "up", "down"]): Arrow direction.
        text (str | None): Optional label text near the arrow tail.
        textstyle (ArrowTextStyle): Text style options (alignment/rotation, and a fallback when
            ``label_font_options`` is None).
        arrow_length_percentage (float | None): Optional arrow length as a percent of axes span
            in the arrow direction. Cannot be combined with ``placement.arrowtail``.
        label_position (tuple[float, float] | None): Optional explicit text-anchor location.
        label_font_options (LabelFontOptions | None): Optional geoplot-style font options; when
            set, these override text color and typography fields from ``textstyle``.
        label_box_options (LabelBoxOptions | None): Optional geoplot-style text-box options.
        placement (ArrowPlacement): Placement options.
        style (LabelArrowStyle): Arrow styling.
        name (str | None): Optional identifier for callers.
    """

    arrowtip: tuple[float, float]
    direction: Literal["right", "left", "up", "down"]
    text: str | None = None
    textstyle: ArrowTextStyle = field(default_factory=ArrowTextStyle)
    arrow_length_percentage: float | None = None
    label_position: tuple[float, float] | None = None
    label_font_options: LabelFontOptions | None = None
    label_box_options: LabelBoxOptions | None = None
    placement: ArrowPlacement = field(default_factory=ArrowPlacement)
    style: LabelArrowStyle = field(default_factory=LabelArrowStyle)
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_arrow_direction(self.direction)
        object.__setattr__(
            self, "arrowtip", _validated_point(self.arrowtip, field_name="_LabelArrowData.arrowtip")
        )

        if self.label_position is not None:
            object.__setattr__(
                self,
                "label_position",
                _validated_point(self.label_position, field_name="_LabelArrowData.label_position"),
            )

        if self.arrow_length_percentage is not None:
            arrow_length_percentage = float(self.arrow_length_percentage)
            if not math.isfinite(arrow_length_percentage):
                raise ValueError("_LabelArrowData.arrow_length_percentage must be finite.")
            if not (0.0 <= arrow_length_percentage <= 100.0):
                raise ValueError("_LabelArrowData.arrow_length_percentage must be in [0, 100].")
            if self.placement.arrowtail is not None:
                raise ValueError(
                    "_LabelArrowData.arrow_length_percentage cannot be set when "
                    "placement.arrowtail is set."
                )
            object.__setattr__(self, "arrow_length_percentage", arrow_length_percentage)


_AnyArrowData: TypeAlias = "_TextArrowData | _LabelArrowData"
