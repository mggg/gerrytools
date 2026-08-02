"""Deferred annotation-arrow rendering for the data plots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import matplotlib.patheffects as patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patheffects import AbstractPathEffect
from matplotlib.text import Text
from matplotlib.transforms import Transform

from gerrytools.colors import resolve_rgba
from gerrytools.plotting._artist_registry import _ArtistRegistry
from gerrytools.plotting.data._gerryplot_dataclasses import (
    ArrowTextStyle,
    _AnyArrowData,
    _LabelArrowData,
    _TextArrowData,
)
from gerrytools.typing import Color

# Owner name used in color-resolution warnings.
_OWNER = "_AnnotationArrowRenderer"


@dataclass(frozen=True, slots=True)
class _PendingTextAlignment:
    bbox_artist: Text
    text_artist: Text
    desired_tip: tuple[float, float]
    transform: Transform
    direction: Literal["right", "left", "up", "down"]


class _VerticalTextArrowBoxStyle:
    """Centered top/bottom triangular-tip boxstyle for vertical text arrows."""

    # Tip geometry as fractions of the boxstyle mutation size.
    _TIP_HEIGHT_SCALE = 0.8
    _TIP_WIDTH_SCALE = 0.28

    def __init__(self, *, direction: Literal["up", "down"], pad: float) -> None:
        self.direction = direction
        self.pad = float(pad)

    def __call__(
        self,
        x0: float,
        y0: float,
        width: float,
        height: float,
        mutation_size: float,
    ) -> Path:
        """Create a box path with a centered triangular tip at top or bottom."""
        pad_pixels = self.pad * mutation_size
        left = x0 - pad_pixels
        bottom = y0 - pad_pixels
        body_width = width + (2.0 * pad_pixels)
        body_height = height + (2.0 * pad_pixels)
        right = left + body_width
        top = bottom + body_height
        center_x = left + (0.5 * body_width)

        tip_height = max(0.0, float(mutation_size) * self._TIP_HEIGHT_SCALE)
        head_overhang = max(0.0, float(mutation_size) * self._TIP_WIDTH_SCALE)

        if self.direction == "up":
            tip_base_y = top
            tip_y = top + tip_height
            vertices = [
                (left, bottom),
                (right, bottom),
                (right, tip_base_y),
                (right + head_overhang, tip_base_y),
                (center_x, tip_y),
                (left - head_overhang, tip_base_y),
                (left, tip_base_y),
                (left, bottom),
            ]
        else:
            tip_base_y = bottom
            tip_y = bottom - tip_height
            vertices = [
                (left, top),
                (right, top),
                (right, tip_base_y),
                (right + head_overhang, tip_base_y),
                (center_x, tip_y),
                (left - head_overhang, tip_base_y),
                (left, tip_base_y),
                (left, top),
            ]

        codes = [Path.MOVETO] + ([Path.LINETO] * (len(vertices) - 2)) + [Path.CLOSEPOLY]
        return Path(vertices, codes)


class _AnnotationArrowRenderer:
    """Internal helper for rendering deferred annotation arrows.

    Decoupled from `GerryPlotBase`: it takes the matplotlib axes, figure, and artist
    registry, and owns its direction/alignment helpers itself.
    """

    def __init__(self, *, ax: Axes, fig: Figure, registry: _ArtistRegistry) -> None:
        self._ax = ax
        self._fig = fig
        self._registry = registry

    # -- public ---------------------------------------------------------

    def render_all(self, arrows: Sequence[_AnyArrowData]) -> None:
        """Render all deferred arrows onto the configured axes."""
        pending: list[_PendingTextAlignment] = []
        for arrow in arrows:
            alignment = self._render_arrow(arrow)
            if alignment is not None:
                pending.append(alignment)

        for _ in range(2):
            if not pending:
                return
            self._fig.canvas.draw()
            moved = False
            for alignment in pending:
                moved |= self._align_text_arrow_tip_to_position(
                    alignment.bbox_artist,
                    desired_tip=alignment.desired_tip,
                    coordinate_transform=alignment.transform,
                    direction=alignment.direction,
                )
                if alignment.text_artist is not alignment.bbox_artist:
                    alignment.text_artist.set_position(alignment.bbox_artist.get_position())
            if not moved:
                return

    # -- direction / alignment helpers (pure, no instance state) --------

    @staticmethod
    def _direction_unit_vector(
        direction: Literal["right", "left", "up", "down"],
    ) -> tuple[float, float]:
        mapping: dict[Literal["right", "left", "up", "down"], tuple[float, float]] = {
            "right": (1.0, 0.0),
            "left": (-1.0, 0.0),
            "up": (0.0, 1.0),
            "down": (0.0, -1.0),
        }
        return mapping[direction]

    @staticmethod
    def _default_text_alignment_for_direction(
        direction: Literal["right", "left", "up", "down"],
    ) -> tuple[Literal["left", "center", "right"], Literal["bottom", "center", "top"]]:
        # Label text anchors just past the tail; alignment must extend the text away from the
        # arrow body (e.g. an "up" arrow hangs its text below the anchor, not up into the shaft).
        mapping: dict[
            Literal["right", "left", "up", "down"],
            tuple[Literal["left", "center", "right"], Literal["bottom", "center", "top"]],
        ] = {
            "right": ("right", "center"),
            "left": ("left", "center"),
            "up": ("center", "top"),
            "down": ("center", "bottom"),
        }
        return mapping[direction]

    @staticmethod
    def _default_text_arrow_boxstyle(
        direction: Literal["right", "left", "up", "down"],
        *,
        pad: float,
    ) -> str | object:
        """The default arrow-shaped boxstyle for a direction.

        Horizontal arrows use matplotlib's built-in ``rarrow``/``larrow`` boxstyles;
        vertical arrows use the custom callable boxstyle, since matplotlib has no
        vertical arrow boxstyle.
        """
        if direction == "right":
            return f"rarrow,pad={pad:g}"
        if direction == "left":
            return f"larrow,pad={pad:g}"
        return _VerticalTextArrowBoxStyle(direction="up" if direction == "up" else "down", pad=pad)

    @staticmethod
    def _directional_extreme_point(
        points: list[tuple[float, float]],
        direction: Literal["right", "left", "up", "down"],
    ) -> tuple[float, float]:
        if direction == "right":
            return max(points, key=lambda point: point[0])
        if direction == "left":
            return min(points, key=lambda point: point[0])
        if direction == "up":
            return max(points, key=lambda point: point[1])
        return min(points, key=lambda point: point[1])

    # -- helpers needing fig / color resolver ---------------------------

    def _direction_display_unit_vector(
        self,
        *,
        origin: tuple[float, float],
        direction: Literal["right", "left", "up", "down"],
        coordinate_transform: Transform,
    ) -> tuple[float, float]:
        unit_x, unit_y = self._direction_unit_vector(direction)
        origin_display = coordinate_transform.transform((origin[0], origin[1]))
        forward_display = coordinate_transform.transform((origin[0] + unit_x, origin[1] + unit_y))

        vector_x = float(forward_display[0] - origin_display[0])
        vector_y = float(forward_display[1] - origin_display[1])
        norm = math.hypot(vector_x, vector_y)
        if norm > 1e-12:
            return (vector_x / norm, vector_y / norm)

        # A zero-size axes can collapse the display vector.
        return (unit_x, unit_y)

    def _shift_point_along_direction_pixels(
        self,
        point: tuple[float, float],
        *,
        direction: Literal["right", "left", "up", "down"],
        signed_pixels: float,
        coordinate_transform: Transform,
    ) -> tuple[float, float]:
        direction_display_x, direction_display_y = self._direction_display_unit_vector(
            origin=point,
            direction=direction,
            coordinate_transform=coordinate_transform,
        )
        start_display = coordinate_transform.transform((point[0], point[1]))
        shifted_display = (
            float(start_display[0]) + (direction_display_x * signed_pixels),
            float(start_display[1]) + (direction_display_y * signed_pixels),
        )
        shifted = coordinate_transform.inverted().transform(shifted_display)
        return (float(shifted[0]), float(shifted[1]))

    def _align_text_arrow_tip_to_position(
        self,
        text_artist: Text,
        *,
        desired_tip: tuple[float, float],
        coordinate_transform: Transform,
        direction: Literal["right", "left", "up", "down"],
    ) -> bool:
        bbox_patch = text_artist.get_bbox_patch()
        if (
            bbox_patch is None
        ):  # pragma: no cover - only possible if the text artist was created without a bbox boxstyle, which cannot happen through the public API
            return False

        vertices_display = bbox_patch.get_transform().transform(bbox_patch.get_path().vertices)
        points: list[tuple[float, float]] = [
            (float(vertex[0]), float(vertex[1])) for vertex in vertices_display
        ]
        if len(points) == 0:  # pragma: no cover - a realized boxstyle bbox path always has vertices
            return False

        current_tip_x, current_tip_y = self._directional_extreme_point(points, direction)
        desired_tip_display = coordinate_transform.transform((desired_tip[0], desired_tip[1]))
        desired_tip_x = float(desired_tip_display[0])
        desired_tip_y = float(desired_tip_display[1])

        delta_x = desired_tip_x - current_tip_x
        delta_y = desired_tip_y - current_tip_y
        if abs(delta_x) < 1e-8 and abs(delta_y) < 1e-8:
            return False

        current_position = text_artist.get_position()
        current_display = coordinate_transform.transform(
            (float(current_position[0]), float(current_position[1]))
        )
        moved_display = (
            float(current_display[0]) + delta_x,
            float(current_display[1]) + delta_y,
        )
        moved_position = coordinate_transform.inverted().transform(moved_display)
        text_artist.set_position((float(moved_position[0]), float(moved_position[1])))
        return True

    @staticmethod
    def _stroke_outline_effects(
        color: Color | None,
        alpha: float | None,
        width: float,
        *,
        field: str,
    ) -> list[AbstractPathEffect] | None:
        """Build stroke-then-normal path effects for a text outline.

        The one shared skip rule: no outline color, or a zero outline width, means no
        path effects at all rather than an invisible zero-width stroke.
        """
        if color is None:
            return None
        if width <= 0:
            return None
        outline_color = resolve_rgba(color, alpha, field=field, owner=_OWNER)
        return [
            patheffects.Stroke(linewidth=float(width), foreground=outline_color),
            patheffects.Normal(),
        ]

    def _annotation_text_outline_effects(
        self,
        textstyle: ArrowTextStyle,
    ) -> list[AbstractPathEffect] | None:
        return self._stroke_outline_effects(
            textstyle.fontoutlinecolor,
            textstyle.fontoutlinealpha,
            textstyle.fontoutlinewidth,
            field="annotation_arrow_text_outlinecolor",
        )

    # -- arrow rendering ------------------------------------------------

    def _render_arrow(self, arrow: _AnyArrowData) -> _PendingTextAlignment | None:
        placement = arrow.placement
        coordinate_system = placement.coordinate_system
        transform = self._ax.transData if coordinate_system == "data" else self._ax.transAxes

        tip_x, tip_y = arrow.arrowtip
        offset_x, offset_y = placement.text_offset

        default_ha, default_va = self._default_text_alignment_for_direction(arrow.direction)
        ha = (
            arrow.textstyle.horizontalalignment
            if arrow.textstyle.horizontalalignment is not None
            else default_ha
        )
        va = (
            arrow.textstyle.verticalalignment
            if arrow.textstyle.verticalalignment is not None
            else default_va
        )

        if isinstance(arrow, _TextArrowData):
            return self._render_text_arrow(
                arrow=arrow,
                transform=transform,
                tip_x=tip_x,
                tip_y=tip_y,
                offset_x=offset_x,
                offset_y=offset_y,
                ha=ha,
                va=va,
            )

        self._render_label_arrow(
            arrow=arrow,
            transform=transform,
            coordinate_system=coordinate_system,
            tip_x=tip_x,
            tip_y=tip_y,
            offset_x=offset_x,
            offset_y=offset_y,
            ha=ha,
            va=va,
        )
        return None

    def _render_text_arrow(
        self,
        *,
        arrow: _TextArrowData,
        transform: Transform,
        tip_x: float,
        tip_y: float,
        offset_x: float,
        offset_y: float,
        ha: Literal["left", "center", "right"],
        va: Literal["bottom", "center", "top"],
    ) -> _PendingTextAlignment:
        style = arrow.style

        text_rotation = (
            float(arrow.textstyle.rotation) if arrow.textstyle.rotation is not None else 0.0
        )
        if style.boxstyle is not None:
            boxstyle: str | object = style.boxstyle
        else:
            boxstyle = self._default_text_arrow_boxstyle(arrow.direction, pad=style.boxpad)

        if isinstance(style.arrowedgecolor, str) and style.arrowedgecolor.lower() == "none":
            edgecolor: str | tuple[float, float, float, float] = "none"
        else:
            edgecolor = resolve_rgba(
                style.arrowedgecolor,
                style.arrowedgealpha,
                field="annotation_arrow_outlinecolor",
                owner=_OWNER,
            )

        text_value = arrow.text if arrow.text is not None else "   "
        text_color = resolve_rgba(
            arrow.textstyle.fontcolor,
            arrow.textstyle.fontalpha,
            field="annotation_arrow_text_color",
            owner=_OWNER,
        )
        face_color = resolve_rgba(
            style.arrowfacecolor,
            style.arrowfacealpha,
            field="annotation_arrow_facecolor",
            owner=_OWNER,
        )

        def draw_text(
            x: float,
            y: float,
            *,
            color: tuple[float, float, float, float],
            rotation: float,
            zorder: float,
            with_box: bool,
        ) -> Text:
            """One arrow text artist; the three call sites differ only in these kwargs."""
            artist = self._ax.text(
                x,
                y,
                text_value,
                transform=transform,
                ha=ha,
                va=va,
                color=color,
                fontsize=arrow.textstyle.fontsize,
                fontweight=arrow.textstyle.fontweight,
                fontstyle=arrow.textstyle.fontstyle,
                fontfamily=arrow.textstyle.fontfamily,
                rotation=rotation,
                clip_on=arrow.placement.clip_on,
                zorder=zorder,
                bbox=(
                    dict(
                        boxstyle=boxstyle,
                        fc=face_color,
                        ec=edgecolor,
                        lw=style.arrowedgewidth,
                    )
                    if with_box
                    else None
                ),
            )
            self._registry.track(artist)
            return artist

        # The arrow box always draws unrotated (rotation 0); when the text itself is rotated,
        # the box and the text must be separate artists so only the glyphs rotate.
        if abs(text_rotation) > 1e-8:
            bbox_artist = draw_text(
                tip_x + offset_x,
                tip_y + offset_y,
                color=(0.0, 0.0, 0.0, 0.0),
                rotation=0.0,
                zorder=arrow.placement.zorder,
                with_box=True,
            )
            text_artist = draw_text(
                tip_x + offset_x,
                tip_y + offset_y,
                color=text_color,
                rotation=text_rotation,
                zorder=float(arrow.placement.zorder) + 0.01,
                with_box=False,
            )
        else:
            text_artist = draw_text(
                tip_x + offset_x,
                tip_y + offset_y,
                color=text_color,
                rotation=0.0,
                zorder=arrow.placement.zorder,
                with_box=True,
            )
            bbox_artist = text_artist

        text_effects = self._annotation_text_outline_effects(arrow.textstyle)
        if text_effects is not None:
            text_artist.set_path_effects(text_effects)
        return _PendingTextAlignment(
            bbox_artist=bbox_artist,
            text_artist=text_artist,
            desired_tip=(tip_x + offset_x, tip_y + offset_y),
            transform=transform,
            direction=arrow.direction,
        )

    def _render_label_arrow(
        self,
        *,
        arrow: _LabelArrowData,
        transform: Transform,
        coordinate_system: Literal["data", "axes fraction"],
        tip_x: float,
        tip_y: float,
        offset_x: float,
        offset_y: float,
        ha: Literal["left", "center", "right"],
        va: Literal["bottom", "center", "top"],
    ) -> None:
        style = arrow.style

        placement = arrow.placement
        unit_x, unit_y = self._direction_unit_vector(arrow.direction)
        if placement.arrowtail is None:
            if arrow.arrow_length_percentage is not None:
                self._fig.canvas.draw()
                axes_bbox = self._ax.get_window_extent()
                if arrow.direction in {"left", "right"}:
                    direction_span_pixels = float(axes_bbox.width)
                else:
                    direction_span_pixels = float(axes_bbox.height)
                arrow_length_pixels = (
                    float(arrow.arrow_length_percentage) / 100.0
                ) * direction_span_pixels
                tail_x, tail_y = self._shift_point_along_direction_pixels(
                    (tip_x, tip_y),
                    direction=arrow.direction,
                    signed_pixels=-arrow_length_pixels,
                    coordinate_transform=transform,
                )
            else:
                tail_x = tip_x - (unit_x * placement.tail_length)
                tail_y = tip_y - (unit_y * placement.tail_length)
        else:
            tail_x, tail_y = placement.arrowtail

        if isinstance(style.arrowedgecolor, str) and style.arrowedgecolor.lower() == "none":
            regular_edgecolor: str | tuple[float, float, float, float] = "none"
        else:
            regular_edgecolor = resolve_rgba(
                style.arrowedgecolor,
                style.arrowedgealpha,
                field="annotation_arrow_outlinecolor",
                owner=_OWNER,
            )

        annotation = self._ax.annotate(
            "",
            xy=(tip_x, tip_y),
            xytext=(tail_x, tail_y),
            xycoords=coordinate_system,
            textcoords=coordinate_system,
            clip_on=placement.clip_on,
            zorder=placement.zorder,
            arrowprops=dict(
                arrowstyle=style.arrowstyle,
                connectionstyle=style.connectionstyle,
                mutation_scale=style.arrowhead_scale,
                shrinkA=style.shrink_a,
                shrinkB=style.shrink_b,
                facecolor=resolve_rgba(
                    style.arrowfacecolor,
                    style.arrowfacealpha,
                    field="annotation_arrow_facecolor",
                    owner=_OWNER,
                ),
                edgecolor=regular_edgecolor,
                linewidth=style.arrowedgewidth,
                linestyle=style.linestyle,
            ),
        )
        self._registry.track(annotation)

        text_value = arrow.text if arrow.text is not None else ""
        if text_value == "":
            return

        if arrow.label_position is None:
            label_anchor_x = tail_x - (unit_x * placement.label_padding)
            label_anchor_y = tail_y - (unit_y * placement.label_padding)
        else:
            label_anchor_x, label_anchor_y = arrow.label_position

        text_x = label_anchor_x + offset_x
        text_y = label_anchor_y + offset_y
        bbox: dict[str, object] | None = (
            None if arrow.label_box_options is None else arrow.label_box_options.to_mpl_bbox()
        )

        if arrow.label_font_options is not None:
            text_effects = self._stroke_outline_effects(
                arrow.label_font_options.outlinecolor,
                None,
                arrow.label_font_options.outlinewidth,
                field="annotation_arrow_label_outlinecolor",
            )
            text_artist = self._ax.text(
                text_x,
                text_y,
                text_value,
                transform=transform,
                ha=ha,
                va=va,
                rotation=arrow.textstyle.rotation if arrow.textstyle.rotation is not None else 0.0,
                clip_on=placement.clip_on,
                zorder=placement.zorder,
                bbox=bbox,
                **arrow.label_font_options.to_mpl_text_kwargs(),
            )
        else:
            text_effects = self._annotation_text_outline_effects(arrow.textstyle)
            text_artist = self._ax.text(
                text_x,
                text_y,
                text_value,
                transform=transform,
                ha=ha,
                va=va,
                rotation=arrow.textstyle.rotation if arrow.textstyle.rotation is not None else 0.0,
                clip_on=placement.clip_on,
                zorder=placement.zorder,
                bbox=bbox,
                color=resolve_rgba(
                    arrow.textstyle.fontcolor,
                    arrow.textstyle.fontalpha,
                    field="annotation_arrow_text_color",
                    owner=_OWNER,
                ),
                fontsize=arrow.textstyle.fontsize,
                fontweight=arrow.textstyle.fontweight,
                fontstyle=arrow.textstyle.fontstyle,
                fontfamily=arrow.textstyle.fontfamily,
            )
        self._registry.track(text_artist)
        text_artist.set_clip_path(self._ax.patch)
        if text_effects is not None:
            text_artist.set_path_effects(text_effects)
