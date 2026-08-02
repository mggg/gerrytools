"""Annotation facade (``add_*`` / ``clear_*``) for :class:`GerryPlotBase`.

Internal module. `_AnnotationApiMixin` carries the public annotation API: line, band, and
arrow adders plus their clear methods. The methods validate and merge styling, then store
records on the `_Annotations` holder that the rebuild flow renders.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting.data._annotations import _Annotations
from gerrytools.plotting.data._gerryplot_dataclasses import (
    ArrowPlacement,
    ArrowTextStyle,
    LabelArrowOptions,
    TextArrowOptions,
    TextArrowStyle,
    _BandData,
    _LabelArrowData,
    _LineData,
    _TextArrowData,
)
from gerrytools.plotting.data.options import BandOptions, LineOptions
from gerrytools.plotting.mpl.label_text_options import LabelStyle
from gerrytools.plotting.utils import (
    UNSET,
    Unset,
    _coerce_real_iter,
    _replace_non_none,
    _replace_with_color_overrides,
)
from gerrytools.typing import Color


class _AnnotationApiMixin:
    """Annotation ``add_*`` / ``clear_*`` API for ``GerryPlotBase``.

    Pure capability mixin: operates on the ``_Annotations`` holder created by the
    ``GerryPlotBase`` constructor and calls ``_LegendMixin``'s named-add legend claim
    for named lines and bands; the declarations below name that contract.
    """

    _annotations: _Annotations

    if TYPE_CHECKING:

        def _claim_legend_if_named(self, name: str | None) -> None: ...

    @deferred_axis_update
    def _add_lines(
        self,
        values: list[float],
        orientation: Literal["vertical", "horizontal"],
        *,
        line_options: LineOptions | None,
        name: str | None,
        **overrides: object,
    ) -> None:
        """Merge styling and store one line annotation for ``orientation``."""
        base = line_options if line_options is not None else LineOptions()
        # The orientation default (3 vertical, 4 horizontal) applies whenever the options
        # object carries no explicit zorder; an explicit zorder kwarg still overrides below.
        if base._zorder_defaulted:
            base = _replace_non_none(base, zorder=3 if orientation == "vertical" else 4)
        style = _replace_with_color_overrides(base, ("linecolor", "linealpha"), **overrides)
        target = (
            self._annotations.vertical_lines
            if orientation == "vertical"
            else self._annotations.horizontal_lines
        )
        target.append(_LineData(values=tuple(values), style=style, name=name))
        self._claim_legend_if_named(name)

    @deferred_axis_update
    def _add_band(
        self,
        low: float,
        high: float,
        orientation: Literal["vertical", "horizontal"],
        *,
        band_options: BandOptions | None,
        name: str | None,
        **overrides: object,
    ) -> None:
        """Merge styling and store one band annotation for ``orientation``."""
        base = band_options if band_options is not None else BandOptions()
        linecolor_defaulted = base._linecolor_defaulted
        # The orientation default (3 vertical, 4 horizontal) applies whenever the options
        # object carries no explicit zorder; an explicit zorder kwarg still overrides below.
        if base._zorder_defaulted:
            base = _replace_non_none(base, zorder=3 if orientation == "vertical" else 4)
        style = _replace_with_color_overrides(
            base,
            ("bandcolor", "bandalpha"),
            bandcolor=overrides["bandcolor"],
            bandalpha=overrides["bandalpha"],
        )
        if linecolor_defaulted and not isinstance(overrides["bandcolor"], Unset):
            # Re-resolve the default boundary from the overridden fill. This also preserves
            # BandOptions' neutral boundary fallback when the new fill is transparent.
            style = dataclasses.replace(style, linecolor=UNSET)
        style = _replace_with_color_overrides(
            style,
            ("linecolor", "linealpha"),
            linecolor=overrides["linecolor"],
            linealpha=overrides["linealpha"],
            linestyle=overrides["linestyle"],
            linewidth=overrides["linewidth"],
            zorder=overrides["zorder"],
        )
        target = (
            self._annotations.vertical_bands
            if orientation == "vertical"
            else self._annotations.horizontal_bands
        )
        # _BandData sorts its bounds, so low/high may arrive in either order.
        target.append(_BandData(lower_bound=low, upper_bound=high, style=style, name=name))
        self._claim_legend_if_named(name)

    def add_vertical_lines(
        self,
        x_values: float | Iterable[float],
        *,
        line_options: LineOptions | None = None,
        linecolor: Color | None | Unset = UNSET,
        linealpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        zorder: int | None = None,
        name: str | None = None,
    ) -> None:
        """Add a vertical line to the figure.

        Args:
            x_values (float | Iterable[float]): The x-value(s) where the vertical line(s) should be
                drawn.
            line_options (LineOptions | None, optional): Optional pre-built styling. Any styling
                kwarg passed explicitly overrides the corresponding field on ``line_options``.
                Defaults to None.
            linecolor (Color | None, optional): The color of the vertical line. Pass ``None`` for
                no line. Defaults to "#cccccc".
            linealpha (float | None, optional): The alpha transparency of the vertical line.
                Defaults to None in which case the alpha from linecolor is used if specified.
            linestyle (str, optional): The linestyle of the vertical line. Defaults to "-".
            linewidth (float, optional): The width of the vertical line. Defaults to 1.0.
            zorder (int, optional): The z-order of the vertical line. Defaults to 3.
            name (str | None, optional): The name of the line for legend purposes. Defaults to None.

        Returns:
            None
        """
        self._add_lines(
            _coerce_real_iter(x_values, field="x_values"),
            "vertical",
            line_options=line_options,
            name=name,
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )

    def add_horizontal_lines(
        self,
        y_values: float | Iterable[float],
        *,
        line_options: LineOptions | None = None,
        linecolor: Color | None | Unset = UNSET,
        linealpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        zorder: int | None = None,
        name: str | None = None,
    ) -> None:
        """Add a horizontal line to the figure.

        Args:
            y_values (float | Iterable[float]): The y-value(s) where the horizontal line(s) should
                be drawn.
            line_options (LineOptions | None, optional): Optional pre-built styling. Any styling
                kwarg passed explicitly overrides the corresponding field on ``line_options``.
                Defaults to None.
            linecolor (Color | None, optional): The color of the horizontal line. Pass ``None``
                for no line. Defaults to "#cccccc".
            linealpha (float | None, optional): The alpha transparency of the horizontal line.
                Defaults to None in which case the alpha from linecolor is used if specified.
            linestyle (str, optional): The linestyle of the horizontal line. Defaults to "-".
            linewidth (float, optional): The width of the horizontal line. Defaults to 1.0.
            zorder (int, optional): The z-order of the horizontal line. Defaults to 4.
            name (str | None, optional): The name of the line for legend purposes. Defaults to None.

        Returns:
            None
        """
        self._add_lines(
            _coerce_real_iter(y_values, field="y_values"),
            "horizontal",
            line_options=line_options,
            name=name,
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )

    def add_vertical_band(
        self,
        x_low: float,
        x_high: float,
        *,
        band_options: BandOptions | None = None,
        bandcolor: Color | None | Unset = UNSET,
        bandalpha: float | None = None,
        linecolor: Color | None | Unset = UNSET,
        linealpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        zorder: int | None = None,
        name: str | None = None,
    ) -> None:
        """Add a vertical band to the figure.

        Args:
            x_low (float): The lower x-value of the vertical band.
            x_high (float): The upper x-value of the vertical band.
            band_options (BandOptions | None, optional): Optional pre-built styling. Any styling
                kwarg passed explicitly overrides the corresponding field on ``band_options``.
                Defaults to None.
            bandcolor (Color | None, optional): The fill color of the band. Pass ``None`` for no
                fill. Defaults to "#cccccc".
            bandalpha (float | None, optional): The alpha transparency of the band. Defaults to
                None.
            linecolor (Color | None, optional): The color of the bounding lines. Pass ``None`` for
                no boundary. When omitted, the line uses ``bandcolor``.
            linealpha (float | None, optional): The alpha transparency of the bounding lines.
                Defaults to None which uses the alpha from linecolor if specified.
            linestyle (str, optional): The linestyle of the bounding lines of the band.
                Defaults to "-".
            linewidth (float, optional): The width of the bounding lines of the band.
                Defaults to 1.0.
            zorder (int, optional): The z-order of the band. Defaults to 3.
            name (str | None, optional): The name of the band for legend purposes. Defaults to None.

        Returns:
            None
        """
        self._add_band(
            x_low,
            x_high,
            "vertical",
            band_options=band_options,
            name=name,
            bandcolor=bandcolor,
            bandalpha=bandalpha,
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )

    def add_horizontal_band(
        self,
        y_low: float,
        y_high: float,
        *,
        band_options: BandOptions | None = None,
        bandcolor: Color | None | Unset = UNSET,
        bandalpha: float | None = None,
        linecolor: Color | None | Unset = UNSET,
        linealpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        zorder: int | None = None,
        name: str | None = None,
    ) -> None:
        """Add a horizontal band to the figure.

        Args:
            y_low (float): The lower y-value of the horizontal band.
            y_high (float): The upper y-value of the horizontal band.
            band_options (BandOptions | None, optional): Optional pre-built styling. Any styling
                kwarg passed explicitly overrides the corresponding field on ``band_options``.
                Defaults to None.
            bandcolor (Color | None, optional): The fill color of the band. Pass ``None`` for no
                fill. Defaults to "#cccccc".
            bandalpha (float | None, optional): The alpha transparency of the band. Defaults to None
            linecolor (Color | None, optional): The color of the bounding lines. Pass ``None`` for
                no boundary. When omitted, the line uses ``bandcolor``.
            linealpha (float | None, optional): The alpha transparency of the bounding lines.
                Defaults to None which uses the alpha from linecolor if specified.
            linestyle (str, optional): The linestyle of the bounding lines of the band.
                Defaults to "-".
            linewidth (float, optional): The width of the bounding lines of the band.
                Defaults to 1.0.
            zorder (int, optional): The z-order of the band. Defaults to 4.
            name (str | None, optional): The name of the band for legend purposes. Defaults to None.

        Returns:
            None
        """
        self._add_band(
            y_low,
            y_high,
            "horizontal",
            band_options=band_options,
            name=name,
            bandcolor=bandcolor,
            bandalpha=bandalpha,
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )

    @deferred_axis_update
    def add_text_arrow(
        self,
        arrowtip: tuple[float, float],
        direction: Literal["right", "left", "up", "down"],
        text: str = "   ",
        *,
        textrotation: float | None = None,
        arrowfacecolor: Color | None | Unset = UNSET,
        arrowfacealpha: float | None = None,
        arrowedgecolor: Color | None | Unset = UNSET,
        arrowedgealpha: float | None = None,
        arrowedgewidth: float | None = None,
        arrowtextstyle: ArrowTextStyle | None = None,
        arrowplacement: ArrowPlacement | None = None,
        arrowstyle: TextArrowStyle | None = None,
        name: str | None = None,
    ) -> None:
        """Add a deferred text-style arrow to the plot.

        This renders via ``Axes.text(..., bbox=...)`` and stores the arrow so it is redrawn
        whenever the plot is rebuilt. The arrow tip is aligned to ``arrowtip`` during rendering.

        Args:
            arrowtip (tuple[float, float]): Arrow-tip coordinate in the selected placement
                coordinate system.
            direction (Literal["right", "left", "up", "down"]): Arrow direction.
            text (str, optional): Text drawn inside the arrow box. Empty strings are normalized
                to ``"   "`` so the arrow still renders. Defaults to ``"   "``.
            textrotation (float | None, optional): Top-level text rotation override in degrees.
                When set, this overrides ``arrowtextstyle.rotation``. Defaults to None.
            arrowfacecolor (Color | None, optional): Optional override for
                ``arrowstyle.arrowfacecolor``. Pass ``None`` for no fill; omission retains
                the style's color.
            arrowfacealpha (float | None, optional): Optional override for
                ``arrowstyle.arrowfacealpha``. Defaults to None.
            arrowedgecolor (Color | None, optional): Optional override for
                ``arrowstyle.arrowedgecolor``. Pass ``None`` for no edge; omission retains
                the style's color.
            arrowedgealpha (float | None, optional): Optional override for
                ``arrowstyle.arrowedgealpha``. Defaults to None.
            arrowedgewidth (float | None, optional): Optional override for
                ``arrowstyle.arrowedgewidth``. Defaults to None.
            arrowtextstyle (ArrowTextStyle | None, optional): Text styling options
                (font, alignment, outline, and rotation). Defaults to None.
            arrowplacement (ArrowPlacement | None, optional): Placement options
                (coordinate system, offsets, clipping, and z-order). Defaults to None.
            arrowstyle (TextArrowStyle | None, optional): Text-arrow box styling
                options. Defaults to None.
            name (str | None, optional): Optional identifier for callers. Defaults to None.

        Returns:
            None
        """
        base_text_style = arrowtextstyle if arrowtextstyle is not None else ArrowTextStyle()
        if textrotation is None:
            arrow_text_style = base_text_style
        else:
            arrow_text_style = dataclasses.replace(base_text_style, rotation=float(textrotation))
        arrow_placement = arrowplacement if arrowplacement is not None else ArrowPlacement()
        style = arrowstyle if arrowstyle is not None else TextArrowStyle()
        merged_textarrowstyle = _replace_with_color_overrides(
            style,
            ("arrowfacecolor", "arrowfacealpha"),
            ("arrowedgecolor", "arrowedgealpha"),
            arrowfacecolor=arrowfacecolor,
            arrowfacealpha=arrowfacealpha,
            arrowedgecolor=arrowedgecolor,
            arrowedgealpha=arrowedgealpha,
            arrowedgewidth=arrowedgewidth,
        )

        text_value = text if text != "" else "   "
        self._annotations.annotation_arrows.append(
            _TextArrowData(
                arrowtip=arrowtip,
                direction=direction,
                text=text_value,
                textstyle=arrow_text_style,
                placement=arrow_placement,
                style=merged_textarrowstyle,
                name=name,
            )
        )

    @deferred_axis_update
    def add_label_arrow(
        self,
        arrowtip: tuple[float, float],
        direction: Literal["right", "left", "up", "down"],
        text: str | None = None,
        *,
        label_position: tuple[float, float] | None = None,
        arrowfacecolor: Color | None | Unset = UNSET,
        arrowfacealpha: float | None = None,
        fontcolor: Color | None | Unset = UNSET,
        textrotation: float | None = None,
        text_options: LabelStyle | None = None,
        arrow_options: LabelArrowOptions | None = None,
        name: str | None = None,
    ) -> None:
        """Add a deferred label-style arrow to the plot.

        This renders a true annotation arrow and an optional separate text label, so the
        arrow length is controlled by tail placement rather than text size.

        Args:
            arrowtip (tuple[float, float]): Arrow-tip coordinate in the selected placement
                coordinate system.
            direction (Literal["right", "left", "up", "down"]): Arrow direction.
            text (str | None, optional): Optional label text near the arrow tail.
                Defaults to None.
            label_position (tuple[float, float] | None, optional): Optional explicit text-anchor
                position in ``arrow_options.placement.coordinate_system``. If None, uses the arrow
                tail plus ``arrow_options.placement.label_padding`` and
                ``arrow_options.placement.text_offset``. Defaults to None.
            arrowfacecolor (Color | None, optional): Optional override for
                ``arrow_options.style.arrowfacecolor``. Pass ``None`` for no fill; omission
                retains the style's color.
            arrowfacealpha (float | None, optional): Optional override for
                ``arrow_options.style.arrowfacealpha``. Defaults to None.
            fontcolor (Color | None, optional): Optional override for the label font color.
                Pass ``None`` for transparent text; omission retains the style's color.
            textrotation (float | None, optional): Optional label rotation in degrees.
                Defaults to None.
            text_options (LabelStyle | None, optional): Label font, outline, and box styling.
                Defaults to None.
            arrow_options (LabelArrowOptions | None, optional): Arrow length, placement, and
                detailed styling. Defaults to None.
            name (str | None, optional): Optional identifier for callers. Defaults to None.

        Returns:
            None
        """
        resolved_arrow_options = arrow_options if arrow_options is not None else LabelArrowOptions()
        arrow_text_style = ArrowTextStyle(rotation=textrotation)
        label_font_options = None if text_options is None else text_options.font
        if not isinstance(fontcolor, Unset):
            resolved_fontcolor: Color | None = "none" if fontcolor is None else fontcolor
            if label_font_options is None:
                arrow_text_style = dataclasses.replace(
                    arrow_text_style, fontcolor=resolved_fontcolor
                )
            else:
                label_font_options = dataclasses.replace(
                    label_font_options, fontcolor=resolved_fontcolor
                )
        merged_labelarrowstyle = _replace_with_color_overrides(
            resolved_arrow_options.style,
            ("arrowfacecolor", "arrowfacealpha"),
            arrowfacecolor=arrowfacecolor,
            arrowfacealpha=arrowfacealpha,
        )

        self._annotations.annotation_arrows.append(
            _LabelArrowData(
                arrowtip=arrowtip,
                direction=direction,
                text=text,
                textstyle=arrow_text_style,
                arrow_length_percentage=resolved_arrow_options.arrow_length,
                label_position=label_position,
                label_font_options=label_font_options,
                label_box_options=(
                    None if text_options is None else text_options.box_for(str(text))
                ),
                placement=resolved_arrow_options.placement,
                style=merged_labelarrowstyle,
                name=name,
            )
        )

    @staticmethod
    def _axis_arrow_geometry(
        axis: Literal["x", "y"],
        *,
        position: float,
        offset: float,
        direction: Literal["right", "left", "up", "down"] | None,
    ) -> tuple[tuple[float, float], Literal["right", "left", "up", "down"]]:
        """Resolve the tip coordinate and direction for an axis direction arrow.

        Raises:
            ValueError: If ``axis`` is invalid, ``position``/``offset`` are not finite, or
                ``direction`` does not run along ``axis``.
        """
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")
        position_value = float(position)
        offset_value = float(offset)
        if not (math.isfinite(position_value) and math.isfinite(offset_value)):
            raise ValueError("position and offset must be finite.")
        if axis == "x":
            resolved_direction = direction if direction is not None else "right"
            if resolved_direction not in ("right", "left"):
                raise ValueError(
                    f"direction for an x-axis arrow must be 'right' or 'left', "
                    f"got {resolved_direction!r}."
                )
            return (position_value, -offset_value), resolved_direction
        resolved_direction = direction if direction is not None else "up"
        if resolved_direction not in ("up", "down"):
            raise ValueError(
                f"direction for a y-axis arrow must be 'up' or 'down', got {resolved_direction!r}."
            )
        return (-offset_value, position_value), resolved_direction

    @staticmethod
    def _axis_arrow_placement(
        arrowplacement: ArrowPlacement | None, *, default_tail_length: float | None
    ) -> ArrowPlacement:
        """Build the axes-fraction placement for an axis direction arrow.

        A caller-supplied placement keeps every field except ``coordinate_system``, which is
        forced to ``"axes fraction"`` because ``position``/``offset`` are axes-fraction values.
        """
        if arrowplacement is not None:
            return dataclasses.replace(arrowplacement, coordinate_system="axes fraction")
        if default_tail_length is not None:
            return ArrowPlacement(
                coordinate_system="axes fraction", tail_length=default_tail_length
            )
        return ArrowPlacement(coordinate_system="axes fraction")

    def add_axis_text_arrow(
        self,
        axis: Literal["x", "y"],
        text: str = "   ",
        *,
        position: float = 0.5,
        offset: float = 0.08,
        direction: Literal["right", "left", "up", "down"] | None = None,
        arrowfacecolor: Color | None | Unset = UNSET,
        arrowfacealpha: float | None = None,
        fontcolor: Color | None | Unset = UNSET,
        textrotation: float | None = None,
        text_options: ArrowTextStyle | None = None,
        arrow_options: TextArrowOptions | None = None,
        name: str | None = None,
    ) -> None:
        """Add a text-style arrow alongside an axis, e.g. to mark the direction of increase.

        The arrow is placed in axes-fraction coordinates just outside the axes: below the
        x-axis or left of the y-axis for a positive ``offset`` (negative flips to the other
        side). The arrow tip lands at ``position`` along the axis and the body extends
        opposite ``direction``, so a right-pointing arrow ends at ``position``. Styling
        and text rotation can be set directly; :class:`ArrowTextStyle` and
        :class:`TextArrowOptions` contain the less common controls.
        :meth:`clear_annotation_arrows` removes these arrows too.

        Args:
            axis (Literal["x", "y"]): Which axis the arrow runs along.
            text (str, optional): Text drawn inside the arrow box. Empty strings are normalized
                to ``"   "`` so the arrow still renders. Defaults to ``"   "``.
            position (float, optional): Arrow-tip position along the axis in axes-fraction
                coordinates (0 is the left/bottom end, 1 the right/top end). Defaults to ``0.5``.
            offset (float, optional): Outward distance from the axis in axes-fraction
                coordinates. Positive places the arrow below the x-axis or left of the y-axis.
                Defaults to ``0.08``.
            direction (Literal["right", "left", "up", "down"] | None, optional): Arrow
                direction; must run along ``axis`` (``"right"``/``"left"`` for x,
                ``"up"``/``"down"`` for y). Defaults to None, meaning ``"right"`` for x and
                ``"up"`` for y.
            arrowfacecolor (Color | None, optional): Optional override for
                ``arrow_options.style.arrowfacecolor``. Pass ``None`` for no fill; omission
                retains the style's color.
            arrowfacealpha (float | None, optional): Optional override for
                ``arrow_options.style.arrowfacealpha``. Defaults to None.
            fontcolor (Color | None, optional): Optional override for the text color. Pass
                ``None`` for transparent text; omission retains the style's color.
            textrotation (float | None, optional): Optional text rotation in degrees.
                Defaults to None.
            text_options (ArrowTextStyle | None, optional): Font, alignment, outline, and
                rotation styling. Defaults to None.
            arrow_options (TextArrowOptions | None, optional): Placement and detailed arrow-box
                styling. The placement coordinate system is always forced to
                ``"axes fraction"``. Defaults to None.
            name (str | None, optional): Optional identifier for callers. Defaults to None.

        Returns:
            None
        """
        arrowtip, resolved_direction = self._axis_arrow_geometry(
            axis, position=position, offset=offset, direction=direction
        )
        resolved_arrow_options = arrow_options if arrow_options is not None else TextArrowOptions()
        resolved_text_options = _replace_with_color_overrides(
            text_options if text_options is not None else ArrowTextStyle(),
            ("fontcolor", "fontalpha"),
            fontcolor=fontcolor,
            rotation=textrotation,
        )
        self.add_text_arrow(
            arrowtip,
            resolved_direction,
            text,
            arrowfacecolor=arrowfacecolor,
            arrowfacealpha=arrowfacealpha,
            arrowtextstyle=resolved_text_options,
            arrowplacement=self._axis_arrow_placement(
                resolved_arrow_options.placement, default_tail_length=None
            ),
            arrowstyle=resolved_arrow_options.style,
            name=name,
        )

    def add_axis_label_arrow(
        self,
        axis: Literal["x", "y"],
        text: str | None = None,
        *,
        position: float = 0.5,
        offset: float = 0.08,
        direction: Literal["right", "left", "up", "down"] | None = None,
        arrowfacecolor: Color | None | Unset = UNSET,
        arrowfacealpha: float | None = None,
        fontcolor: Color | None | Unset = UNSET,
        textrotation: float | None = None,
        text_options: LabelStyle | None = None,
        arrow_options: LabelArrowOptions | None = None,
        name: str | None = None,
    ) -> None:
        """Add a label-style arrow alongside an axis, e.g. to mark the direction of increase.

        The arrow is placed in axes-fraction coordinates just outside the axes: below the
        x-axis or left of the y-axis for a positive ``offset`` (negative flips to the other
        side). The arrow tip lands at ``position`` along the axis, the tail extends opposite
        ``direction`` (``arrow_options.placement.tail_length`` in axes-fraction units, default
        ``0.04``), and the optional ``text`` sits near the tail. Common colors and text rotation
        can be set directly; :class:`LabelStyle` and :class:`LabelArrowOptions` contain the less
        common controls. :meth:`clear_annotation_arrows` removes these arrows too.

        Args:
            axis (Literal["x", "y"]): Which axis the arrow runs along.
            text (str | None, optional): Optional label text near the arrow tail.
                Defaults to None.
            position (float, optional): Arrow-tip position along the axis in axes-fraction
                coordinates (0 is the left/bottom end, 1 the right/top end). Defaults to ``0.5``.
            offset (float, optional): Outward distance from the axis in axes-fraction
                coordinates. Positive places the arrow below the x-axis or left of the y-axis.
                Defaults to ``0.08``.
            direction (Literal["right", "left", "up", "down"] | None, optional): Arrow
                direction; must run along ``axis`` (``"right"``/``"left"`` for x,
                ``"up"``/``"down"`` for y). Defaults to None, meaning ``"right"`` for x and
                ``"up"`` for y.
            arrowfacecolor (Color | None, optional): Optional override for
                ``arrow_options.style.arrowfacecolor``. Pass ``None`` for no fill; omission
                retains the style's color.
            arrowfacealpha (float | None, optional): Optional override for
                ``arrow_options.style.arrowfacealpha``. Defaults to None.
            fontcolor (Color | None, optional): Optional override for the label font color.
                Pass ``None`` for transparent text; omission retains the style's color.
            textrotation (float | None, optional): Optional label rotation in degrees.
                Defaults to None.
            text_options (LabelStyle | None, optional): Label font, outline, and box styling.
                Defaults to None.
            arrow_options (LabelArrowOptions | None, optional): Arrow length, placement, and
                detailed styling. The placement coordinate system is always forced to
                ``"axes fraction"``. Defaults to None.
            name (str | None, optional): Optional identifier for callers. Defaults to None.

        Returns:
            None
        """
        arrowtip, resolved_direction = self._axis_arrow_geometry(
            axis, position=position, offset=offset, direction=direction
        )
        resolved_arrow_options = arrow_options if arrow_options is not None else LabelArrowOptions()
        self.add_label_arrow(
            arrowtip,
            resolved_direction,
            text,
            arrowfacecolor=arrowfacecolor,
            arrowfacealpha=arrowfacealpha,
            fontcolor=fontcolor,
            textrotation=textrotation,
            text_options=text_options,
            arrow_options=dataclasses.replace(
                resolved_arrow_options,
                placement=self._axis_arrow_placement(
                    resolved_arrow_options.placement, default_tail_length=0.04
                ),
            ),
            name=name,
        )

    @deferred_axis_update
    def clear_annotation_arrows(self) -> None:
        """Clear all annotation arrows from the figure."""
        self._annotations.clear_annotation_arrows()

    @deferred_axis_update
    def clear_verticals(self) -> None:
        """Clear all vertical lines and bands from the figure."""
        self._annotations.clear_verticals()

    @deferred_axis_update
    def clear_horizontals(self) -> None:
        """Clear all horizontal lines and bands from the figure."""
        self._annotations.clear_horizontals()
