"""`_MarkerLayer` — point markers with optional labels."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from typing import Any

import matplotlib.patheffects as patheffects
from geopandas import GeoSeries
from matplotlib.axes import Axes

from gerrytools.colors import resolve_color_and_alpha, resolve_rgba
from gerrytools.plotting.geometry._layers._base import _to_target_crs
from gerrytools.plotting.mpl.label_text_options import (
    LabelBoxOptions,
    LabelFontOptions,
    LabelStyle,
)
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import CRSLike


def _normalize_label_key(value: object) -> str:
    """Normalize a label or per-label-override key for matching.

    District labels arrive as ints, zero-padded strings, or plain strings depending on the
    source column; ``1``, ``"1"``, and ``"01"`` all normalize to ``"1"`` so per-label
    ``adjustments`` / ``fontsize`` mappings match regardless of which form the caller used.
    """
    text = str(value)
    try:
        numeric = Decimal(text)
    except InvalidOperation:
        return text
    if numeric.is_finite() and numeric == numeric.to_integral_value():
        return str(int(numeric))
    return text


@dataclass(frozen=True, slots=True)
class _MarkerLayer:
    """A layer of point markers with optional labels.

    Attributes:
        point_geometries (GeoSeries): A GeoSeries of Point geometries for the markers.
        labels (Sequence[str] | None): Optional labels for each marker.
        marker_options (PointMarkerOptions): Marker style settings. Uses default constructor if not
            provided.
        show_labels (bool): Whether to show labels on the markers. Default is True.
        label_font_options (LabelFontOptions): Font options for the labels. Uses default
            constructor if not provided.
        zorder (int): Z-order for rendering. Default is 2.
    """

    point_geometries: GeoSeries

    # Optional labels (same length as point_geometries)
    labels: Sequence[str] | None = None

    # Marker style (shared across the layer)
    marker_options: PointMarkerOptions = field(default_factory=PointMarkerOptions)

    # Label style (centered in marker)
    show_labels: bool = True
    # When set, the label style supersedes the font/box fields and may vary the box per label
    # (e.g. equalizing badge circle diameters).
    label_style: LabelStyle | None = None
    label_font_options: LabelFontOptions = field(default_factory=LabelFontOptions)
    label_box_options: LabelBoxOptions = field(default_factory=LabelBoxOptions)
    # Per-label placement and size tweaks, keyed by (normalized) label. Adjustments are
    # (dx, dy) in the plot's data units and move the label text, not the marker.
    label_adjustments: dict | None = None
    label_fontsize: dict | float | None = None
    zorder: int = 2

    def __post_init__(self) -> None:
        if self.point_geometries is None:
            raise TypeError("MarkerLayer requires `point_geometries` (a GeoSeries of Points).")

        if self.labels is not None and len(self.labels) != len(self.point_geometries):
            raise ValueError("`labels` must have the same length as `point_geometries`.")

    def render(
        self,
        ax: Axes,
        *,
        target_crs: CRSLike | None = None,
    ) -> list:
        """Render this layer onto the given Axes.

        Args:
            ax (Axes): The Axes to render onto.
            target_crs (CRSLike | None, optional): The target CRS to reproject geometries to.
                Defaults to None.

        Returns:
            list[Artist]: The matplotlib artists added to ``ax`` by this layer.
        """
        point_geometries = _to_target_crs(self.point_geometries, target_crs)

        x_coordinates = point_geometries.x.to_numpy()
        y_coordinates = point_geometries.y.to_numpy()

        # PointMarkerOptions already returns RGBA colors with alpha baked in. Typed as Any
        # values so the dict can be splatted into Axes.plot's typed keyword surface.
        marker_kwargs: dict[str, Any] = {**self.marker_options.to_mpl_settings_dict()}
        marker_kwargs.pop("zorder", None)

        artists: list = []
        if not self.show_labels or self.labels is None:
            marker_lines = ax.plot(
                x_coordinates,
                y_coordinates,
                linestyle="None",
                zorder=int(self.zorder),
                **marker_kwargs,
            )
            artists.extend(marker_lines)
        else:
            font_options = (
                self.label_style.font if self.label_style is not None else self.label_font_options
            )
            outline_color = resolve_rgba(
                font_options.outlinecolor,
                field="outlinecolor",
                owner="MarkerLayer",
            )
            text_effects: list[patheffects.AbstractPathEffect] = [
                patheffects.Stroke(
                    linewidth=float(font_options.outlinewidth),
                    foreground=outline_color,
                ),
                patheffects.Normal(),
            ]

            text_color, text_alpha = resolve_color_and_alpha(
                font_options.fontcolor,
                alpha=font_options.fontalpha,
            )

            adjustments_by_label = {
                _normalize_label_key(key): value
                for key, value in (self.label_adjustments or {}).items()
            }
            if isinstance(self.label_fontsize, dict):
                fontsizes = {
                    _normalize_label_key(key): value for key, value in self.label_fontsize.items()
                }
            else:
                fontsizes = {}

            for x_value, y_value, label_text in zip(x_coordinates, y_coordinates, self.labels):
                marker_lines = ax.plot(
                    x_value,
                    y_value,
                    linestyle="None",
                    zorder=int(self.zorder),
                    **marker_kwargs,
                )
                artists.extend(marker_lines)

                if self.label_style is not None:
                    box_options = self.label_style.box_for(str(label_text))
                    bbox = box_options.to_mpl_bbox() if box_options is not None else None
                else:
                    bbox = self.label_box_options.to_mpl_bbox()

                key = _normalize_label_key(label_text)
                dx, dy = adjustments_by_label.get(key, (0.0, 0.0))
                label_font = font_options
                if key in fontsizes:
                    label_font = replace(font_options, fontsize=float(fontsizes[key]))
                elif isinstance(self.label_fontsize, (int, float)):
                    label_font = replace(font_options, fontsize=float(self.label_fontsize))
                text_artist = ax.text(
                    float(x_value) + float(dx),
                    float(y_value) + float(dy),
                    str(label_text),
                    ha="center",
                    va="center",
                    zorder=int(self.zorder),
                    bbox=bbox,
                    clip_on=True,
                    **label_font.to_mpl_text_kwargs(),
                )
                text_artist.set_clip_path(ax.patch)
                text_artist.set_path_effects(text_effects)
                artists.append(text_artist)

        return artists
