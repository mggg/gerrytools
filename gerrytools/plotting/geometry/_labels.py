"""Deferred geography labels: the options bundle, request record, queueing, and rendering.

Label layers are drawn after limits are applied so representative points fall inside the
final view. The plot classes accumulate `_LabelRequest`s via `_queue_label_request` and the
rebuild flow renders them with `_draw_deferred_labels`.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from shapely.geometry import Point, box

from gerrytools.plotting._artist_registry import _ArtistRegistry
from gerrytools.plotting.geometry._layers._base import _to_target_crs
from gerrytools.plotting.geometry._layers._marker import _MarkerLayer, _normalize_label_key
from gerrytools.plotting.mpl.label_text_options import (
    LabelBoxOptions,
    LabelFontOptions,
    LabelStyle,
    resolve_label_style,
)
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import CategoryKey, CRSLike

# Marker options that draw nothing, for layers that exist only to carry labels.
_INVISIBLE_MARKER = PointMarkerOptions(
    markerfacecolor="none",
    markerfacealpha=0.0,
    marker="o",
    markersize=0.0,
    markeredgecolor="none",
    markeredgealpha=0.0,
    markeredgewidth=0.0,
)

# Default font for geography labels when the caller gives neither font options nor a label style.
_DEFAULT_LABEL_FONT = LabelFontOptions(
    fontcolor="black",
    fontsize=4,
    fontweight="roman",
    outlinecolor="grey",
    outlinewidth=0.2,
)


@dataclass(frozen=True)
class LabelOptions:
    """Bundled styling and placement options for geometry-plot labels.

    One instance travels through every labeled layer method (``add_outline_layer``,
    ``add_highlight_layer``, ``add_marker_layer``, ``add_label_layer``,
    ``add_districting_plan_layer``, and the ``DotDensityPlot`` constructor) as the
    single ``label_options`` argument.

    Attributes:
        label_style (LabelStyle | str | None): A ``LabelStyle`` or the name of a registered style
            (see ``LABEL_STYLES``, e.g. ``"badge"`` for district-number badges or ``"halo"``
            for outlined text). Mutually exclusive with ``font_options`` / ``box_options``.
            A style name resolves to its ``LabelStyle`` at construction. Defaults to None.
        font_options (LabelFontOptions | None): Font options for the labels. When None (and
            no ``label_style`` is given) the plot method's default font applies. Defaults to None.
        box_options (LabelBoxOptions | None): Box options for the labels. When None the box
            is disabled (unless the plot method supplies its own default). Defaults to None.
        adjustments (dict | None): Per-label position tweaks, mapping a label to a
            ``(dx, dy)`` offset in the plot's data units. Keys match labels regardless of
            int/str/zero-padded form. Defaults to None.
        fontsize (dict | float | None): Per-label font-size overrides (a mapping from label
            to size), or one size for every label. Defaults to None.
        exclude (Sequence[CategoryKey] | None): Labels to exclude from labeling, applied to
            every labeled layer (dissolved outline/highlight/plan labels as well as marker
            and label layers, where the excluded points are dropped too). Matching is
            dtype-insensitive: ``1``, ``"1"``, and ``"01"`` refer to the same label.
            Defaults to None.

    Raises:
        ValueError: If ``label_style`` is combined with ``font_options`` / ``box_options``, or
            the style name is unknown.
    """

    label_style: LabelStyle | str | None = None
    font_options: LabelFontOptions | None = None
    box_options: LabelBoxOptions | None = None
    adjustments: dict | None = None
    fontsize: dict | float | None = None
    exclude: Sequence[CategoryKey] | None = None

    def __post_init__(self) -> None:
        if self.label_style is not None:
            if self.font_options is not None or self.box_options is not None:
                raise ValueError(
                    "Pass either `label_style` or explicit `font_options` / `box_options`, "
                    "not both.",
                )
            object.__setattr__(
                self,
                "label_style",
                resolve_label_style(self.label_style),
            )

    @property
    def resolved_label_style(self) -> LabelStyle | None:
        """Return the resolved ``LabelStyle``, or None when no label style was given."""
        # __post_init__ has already replaced any style name with its LabelStyle.
        label_style = self.label_style
        assert not isinstance(label_style, str)
        return label_style


@dataclass(frozen=True, slots=True)
class _LabelRequest:
    gdf: GeoDataFrame
    label_column: str
    options: LabelOptions
    label_format_fn: Callable[[CategoryKey], str] | None = None
    zorder: int = 100
    dissolved: bool = False


def _merge_label_style_arg(
    label_style: LabelStyle | str | None,
    label_options: LabelOptions | None,
) -> LabelOptions | None:
    """Fold the top-level ``label_style=`` shorthand into the ``label_options`` bundle.

    Raises:
        ValueError: If both the shorthand and ``label_options.label_style`` are given.
    """
    if label_style is None:
        return label_options
    if label_options is None:
        return LabelOptions(label_style=label_style)
    if label_options.label_style is not None:
        raise ValueError("Pass `label_style` or `label_options.label_style`, not both.")
    return replace(label_options, label_style=label_style)


def _label_keep_mask(labels: Sequence[object], exclude: Sequence[CategoryKey]) -> list[bool]:
    """Per-label keep flags for ``LabelOptions.exclude`` matching.

    Shared by every label-add boundary so exclusion uses one key normalization:
    ``1``, ``"1"``, and ``"01"`` all refer to the same label.
    """
    excluded = {_normalize_label_key(value) for value in exclude}
    return [_normalize_label_key(label) not in excluded for label in labels]


def _queue_label_request(
    requests: list[_LabelRequest],
    *,
    gdf: GeoDataFrame,
    label_column: str,
    options: LabelOptions | None,
    label_format_fn: Callable[[CategoryKey], str] | None = None,
    zorder: int,
    dissolved: bool = False,
) -> None:
    """Append one label request, applying the exclude filter and default-font fallback."""
    options = options if options is not None else LabelOptions()

    if options.exclude:
        keep = _label_keep_mask(gdf[label_column].tolist(), options.exclude)
        gdf = GeoDataFrame(gdf[keep])

    if options.font_options is None and options.label_style is None:
        options = replace(options, font_options=_DEFAULT_LABEL_FONT)

    requests.append(
        _LabelRequest(
            gdf=gdf,
            label_column=label_column,
            options=options,
            label_format_fn=label_format_fn,
            zorder=zorder,
            dissolved=dissolved,
        )
    )


def _draw_deferred_labels(
    requests: Sequence[_LabelRequest],
    *,
    ax: Axes,
    target_crs: CRSLike | None,
    artists: _ArtistRegistry,
) -> dict[str, Point]:
    """Draw all deferred labels onto ``ax`` and return their positions.

    Args:
        requests (Sequence[_LabelRequest]): The accumulated label requests.
        ax (Axes): The axes to draw onto; its current view clips the labels.
        target_crs (CRSLike | None): The plot's target CRS.
        artists (_ArtistRegistry): Registry that tracks the created label artists.

    Returns:
        dict[str, Point]: A dictionary mapping label text to Point objects. If separate label
            requests render the same text, the last request's position is returned.
    """
    label_positions: dict[str, Point] = {}
    if not requests:
        return label_positions

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    clip_geom = box(min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax))

    for req in requests:
        # One label per dissolved part.
        dissolved = (
            req.gdf
            if req.dissolved
            else GeoDataFrame(req.gdf.dissolve(by=req.label_column).reset_index())
        )

        # Match plot CRS
        dissolved = _to_target_crs(dissolved, target_crs)

        # Clip to current view
        clipped = dissolved.geometry.intersection(clip_geom)
        keep = (~clipped.isna()) & (~clipped.is_empty)
        if not keep.any():
            continue

        # Assign through the active geometry column's name; a literal "geometry" key would
        # add a new column and leave a renamed active column unclipped.
        dissolved = GeoDataFrame(dissolved.loc[keep].copy())
        dissolved[dissolved.geometry.name] = clipped.loc[keep]

        # Representative points inside the clipped geometry
        pts = dissolved.representative_point()

        format_fn = req.label_format_fn if req.label_format_fn is not None else str
        labels = [str(format_fn(raw)) for raw in dissolved[req.label_column].tolist()]
        seen: set[str] = set()
        duplicates = set()
        for label in labels:
            if label in seen:
                duplicates.add(label)
            seen.add(label)
        if duplicates:
            formatted = ", ".join(repr(label) for label in sorted(duplicates))
            raise ValueError(f"Computed label text must be unique; duplicates: {formatted}.")

        # Defaults (a label style, when present, supersedes the font/box options)
        options = req.options
        font = options.font_options if options.font_options is not None else LabelFontOptions()
        boxopt = (
            options.box_options
            if options.box_options is not None
            else LabelBoxOptions(enabled=False)
        )

        # An ephemeral label-only marker layer, rendered immediately
        tmp = _MarkerLayer(
            point_geometries=pts,
            labels=labels,
            marker_options=_INVISIBLE_MARKER,
            show_labels=True,
            label_style=options.resolved_label_style,
            label_adjustments=options.adjustments,
            label_fontsize=options.fontsize,
            label_font_options=font,
            label_box_options=boxopt,
            zorder=req.zorder,
        )
        label_artists = tmp.render(ax, target_crs=target_crs)
        if label_artists:
            artists.track(label_artists)
        label_positions.update(
            {label: Point(pt.x, pt.y) for label, pt in zip(labels, pts.geometry.tolist())}
        )
    return label_positions
