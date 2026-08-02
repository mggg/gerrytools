import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from gerrytools.colors import convert_color_to_hexa_or_none
from gerrytools.typing import Color, HexColor


@dataclass(frozen=True)
class SubwaySignOptions:
    """Options for subway sign plotting.

    Attributes:
        radius (float): Radius of each subway sign circle. Default is 0.3.
        edgecolor (Color | None): Edge color of the subway sign circles. None removes
            the edge. Default is "black".
        linewidth (float): Line width of the subway sign circle edges. Default is 1.5.
        fontsize (float): Font size of the label text. Default is 14.
        fontweight (str): Font weight of the label text. Default is "bold".
        fontcolor (Color | None): Font color of the label text. None makes the text
            transparent. Default is "white".
        fontoutlinewidth (float): Width of the outline around the label text. Default is 2.
        horizontalgap (float | None): Horizontal gap between subway signs. If None,
            defaults to 0.3 * radius. Default is None.
        verticalgap (float | None): Vertical gap between subway signs. If None,
            defaults to 0.3 * radius. Default is None.
        padding (float | None): Padding around the entire layout. If None,
            defaults to 0.2 * radius. Default is None.
        raggededge (Literal["first", "last"]): Whether the incomplete band is placed at the start
            or end of the layout. Default is "last".
    """

    radius: float = 0.3
    edgecolor: Color | None = "black"
    linewidth: float = 1.5

    fontsize: float = 14
    fontweight: str = "bold"
    fontcolor: Color | None = "white"
    fontoutlinewidth: float = 2

    horizontalgap: float | None = None
    verticalgap: float | None = None
    padding: float | None = None

    raggededge: Literal["first", "last"] = "last"


def _validate_subway_settings(
    *,
    colors: list[Color | None],
    labels: list[str],
    orientation: Literal["vertical", "horizontal"],
    n_bands: int | None,
    max_items_per_band: int | None,
    sign_options: SubwaySignOptions,
) -> None:
    """Validate settings for subway sign plotting.

    Args:
        colors (list[Color | None]): Sign face colors. None removes a sign's fill.
        labels (list[str]): Sign text labels.
        orientation (Literal["vertical", "horizontal"]): Layout orientation.
        n_bands (int | None): Number of rows/columns, depending on orientation.
            Defaults to None.
        max_items_per_band (int | None): Max items per row/column, depending on orientation.
            Defaults to None.
        sign_options (SubwaySignOptions): Rendering and spacing options.

    Returns:
        None

    Raises:
        ValueError: If any settings are invalid.
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("`orientation` must be either 'vertical' or 'horizontal'.")
    if sign_options.raggededge not in ("first", "last"):
        raise ValueError("`sign_options.raggededge` must be 'first' or 'last'.")
    if n_bands is not None and max_items_per_band is not None:
        raise ValueError("Only one of `n_bands` and `max_items_per_band` may be set.")
    if len(colors) != len(labels):
        raise ValueError("`colors` and `labels` must have the same length.")
    if len(labels) == 0:
        raise ValueError("Nothing to draw: `labels` is empty.")
    if sign_options.radius <= 0:
        raise ValueError("`sign_options.radius` must be > 0.")
    if n_bands is not None and n_bands <= 0:
        raise ValueError("`n_bands` must be > 0.")
    if max_items_per_band is not None and max_items_per_band <= 0:
        raise ValueError("`max_items_per_band` must be > 0.")


@dataclass(frozen=True)
class _SubwayPlotLayout:
    """Layout information for subway sign plotting.

    Attributes:
        item_count (int): Total number of subway signs.
        radius (float): Radius of each subway sign circle.
        diameter (float): Diameter of each subway sign circle.
        horizontalgap (float): Horizontal gap between subway signs.
        verticalgap (float): Vertical gap between subway signs.
        padding (float): Padding around the entire layout.
        row_count (int): Number of rows in the layout.
        column_count (int): Number of columns in the layout.
    """

    item_count: int
    radius: float
    diameter: float
    horizontalgap: float
    verticalgap: float
    padding: float
    row_count: int
    column_count: int


def _determine_offsets_and_counts(
    *,
    labels: list[str],
    orientation: Literal["vertical", "horizontal"] = "horizontal",
    n_bands: int | None = None,
    max_items_per_band: int | None = None,
    sign_options: SubwaySignOptions,
) -> _SubwayPlotLayout:
    """Determine grid dimensions and spacing for subway-sign layout.

    Args:
        labels (list[str]): Sign labels to render.
        orientation (Literal["vertical", "horizontal"], optional): Primary layout direction.
            Defaults to ``"horizontal"``.
        n_bands (int | None, optional): Number of bands (rows for horizontal, columns for
            vertical). Defaults to None.
        max_items_per_band (int | None, optional): Maximum items per band (columns for
            horizontal, rows for vertical). Defaults to None.
        sign_options (SubwaySignOptions): Rendering and spacing options.

    Returns:
        _SubwayPlotLayout: Derived layout geometry and spacing values.
    """
    item_count = len(labels)
    radius = float(sign_options.radius)
    diameter = 2.0 * radius

    horizontalgap = (
        sign_options.horizontalgap if sign_options.horizontalgap is not None else 0.3 * radius
    )
    verticalgap = sign_options.verticalgap if sign_options.verticalgap is not None else 0.3 * radius
    padding = sign_options.padding if sign_options.padding is not None else 0.2 * radius

    # Work in band terms (a band is a row when horizontal, a column when vertical), then
    # transpose once at the end.
    if max_items_per_band is not None:
        band_size = min(int(max_items_per_band), item_count)
        band_count = math.ceil(item_count / band_size)
    elif n_bands is not None:
        band_count = int(n_bands)
        band_size = math.ceil(item_count / band_count)
        # More requested bands than band_size can fill would leave blank bands and a
        # nonpositive ragged size; clamp so 1 <= ragged_size <= band_size always holds.
        band_count = math.ceil(item_count / band_size)
    else:
        band_count = 1
        band_size = item_count

    if orientation == "horizontal":
        row_count, column_count = band_count, band_size
    else:
        row_count, column_count = band_size, band_count

    return _SubwayPlotLayout(
        item_count=item_count,
        radius=radius,
        diameter=diameter,
        horizontalgap=horizontalgap,
        verticalgap=verticalgap,
        padding=padding,
        row_count=row_count,
        column_count=column_count,
    )


def _band_position(
    linear_index: int,
    *,
    item_count: int,
    band_count: int,
    band_size: int,
    ragged_first: bool,
) -> tuple[int, int]:
    """Return ``(band_index, index_in_band)`` for a linear index.

    Bands fill in order with ``band_size`` items each; when the item count does not fill the
    grid, the partial band is the first band when ``ragged_first`` else the last (where plain
    ``divmod`` already leaves it).
    """
    ragged_size = item_count - (band_count - 1) * band_size
    if ragged_first and ragged_size != band_size:
        if linear_index < ragged_size:
            return 0, linear_index
        band_index, index_in_band = divmod(linear_index - ragged_size, band_size)
        return band_index + 1, index_in_band
    return divmod(linear_index, band_size)


def _normalize_colors_and_adjust_item_order(
    *,
    colors: list[Color | None],
    labels: list[str],
    reverse_display_order: bool,
) -> list[tuple[HexColor, str]]:
    """Normalize colors to hex format and optionally reverse the display order.

    Reversing display order reverses the full flattened sequence of signs; the drawing pass
    then reflows that sequence into the same band layout (including which side the ragged
    edge sits on). A single band therefore simply reverses, and a multi-band grid equals
    reversing the flat list and re-banding it.

    Args:
        colors (list[Color | None]): List of colors for each sign. None removes a sign's
            fill.
        labels (list[str]): List of labels for each sign.
        reverse_display_order (bool): Whether to reverse the display order of signs.
    """
    items: list[tuple[HexColor, str]] = [
        (convert_color_to_hexa_or_none(c), label) for c, label in zip(colors, labels)
    ]
    if reverse_display_order:
        items.reverse()
    return items


def _draw_sign(
    *,
    axes: Axes,
    index: int,
    face_color: Color | None,
    label_text: str,
    sign_options: SubwaySignOptions,
    orientation: Literal["vertical", "horizontal"],
    item_count: int,
    row_count: int,
    column_count: int,
    x_step: float,
    y_step: float,
    radius: float,
    text_outline_effects: Sequence[patheffects.AbstractPathEffect],
) -> None:
    """Draw a single subway sign on the given axes.

    Args:
        axes (plt.Axes): The axes to draw on.
        index (int): The linear index of the sign.
        face_color (Color | None): The face color of the sign. None removes the fill.
        label_text (str): The label text of the sign.
        sign_options (SubwaySignOptions): Options for sign appearance.
        orientation (Literal["vertical", "horizontal"]): The layout orientation.
        item_count (int): Total number of signs.
        row_count (int): Number of rows in the layout.
        column_count (int): Number of columns in the layout.
        x_step (float): The horizontal step size between sign centers.
        y_step (float): The vertical step size between sign centers.
        radius (float): The radius of the sign circle.
        text_outline_effects (Sequence[patheffects.AbstractPathEffect]): Path effects for
            outlining the text.
    """
    normalized_face_color = mcolors.to_rgba(convert_color_to_hexa_or_none(face_color))

    horizontal = orientation == "horizontal"
    band_count = row_count if horizontal else column_count
    band_size = column_count if horizontal else row_count
    ragged_first = sign_options.raggededge == "first"

    band_index, index_in_band = _band_position(
        index,
        item_count=item_count,
        band_count=band_count,
        band_size=band_size,
        ragged_first=ragged_first,
    )

    # The one point orientation matters: transpose band coordinates to (row, column).
    if horizontal:
        row_index, col_index = band_index, index_in_band
    else:
        row_index, col_index = index_in_band, band_index

    x_center = col_index * x_step + radius
    y_center = (row_count - 1 - row_index) * y_step + radius

    # Center a partial band by shifting it half the missing width along its own axis.
    ragged_size = item_count - (band_count - 1) * band_size
    ragged_band_index = 0 if ragged_first else band_count - 1
    if ragged_size != band_size and band_index == ragged_band_index:
        shift = 0.5 * (band_size - ragged_size)
        if horizontal:
            x_center += shift * x_step
        else:
            y_center -= shift * y_step

    circle_patch = Circle(
        (x_center, y_center),
        radius=radius,
        facecolor=normalized_face_color,
        edgecolor=sign_options.edgecolor,
        linewidth=sign_options.linewidth,
    )
    axes.add_patch(circle_patch)

    text_artist = axes.text(
        x_center,
        y_center,
        str(label_text),
        ha="center",
        va="center",
        color=sign_options.fontcolor,
        fontsize=sign_options.fontsize,
        fontweight=sign_options.fontweight,
    )
    text_artist.set_path_effects(list(text_outline_effects))


def subway_signs(
    colors: list[Color | None],
    labels: list[str],
    *,
    ax: Axes | None = None,
    orientation: Literal["vertical", "horizontal"] = "horizontal",
    n_bands: int | None = None,
    max_items_per_band: int | None = None,
    reverse_display_order: bool = False,
    sign_options: SubwaySignOptions | None = None,
    filepath: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw a grid of colored 'subway signs' with labels.

    Args:
        colors (list[Color | None]): List of colors for each sign. None removes a sign's
            fill.
        labels (list[str]): List of labels for each sign.
        ax (Axes | None, optional): Axes to draw on. Creates a new figure when omitted.
        orientation (Literal["vertical", "horizontal"], optional): Orientation of the layout.
            Defaults to "horizontal".
        n_bands (int | None, optional): Number of bands (rows or columns) in the layout.
            If None, determined by `max_items_per_band` or defaults to 1 band.
            Defaults to None.
        max_items_per_band (int | None, optional): Maximum number of items per band (row or column).
            If None, determined by `n_bands` or defaults to all items in one band.
            Defaults to None.
        reverse_display_order (bool, optional): Whether to reverse the order in which the
            signs are displayed. The full flattened sequence of signs is reversed and reflowed
            into the same band layout, so the general layout (i.e. which side the ragged edge
            appears on) remains the same. Defaults to False.
        sign_options (SubwaySignOptions | None, optional): Options for sign appearance.
            Defaults to None.
        filepath (str | None, optional): If provided, saves the figure to this path.
            Defaults to None.

    Returns:
        tuple[Figure, Axes]: The figure and axes containing the signs.
    """
    if sign_options is None:
        sign_options = SubwaySignOptions()
    sign_options = replace(
        sign_options,
        edgecolor=convert_color_to_hexa_or_none(sign_options.edgecolor),
        fontcolor=convert_color_to_hexa_or_none(sign_options.fontcolor),
    )

    _validate_subway_settings(
        colors=colors,
        labels=labels,
        orientation=orientation,
        n_bands=n_bands,
        max_items_per_band=max_items_per_band,
        sign_options=sign_options,
    )

    layout = _determine_offsets_and_counts(
        labels=labels,
        orientation=orientation,
        n_bands=n_bands,
        max_items_per_band=max_items_per_band,
        sign_options=sign_options,
    )

    x_step = layout.diameter + layout.horizontalgap
    y_step = layout.diameter + layout.verticalgap

    layout_width = layout.column_count * x_step - layout.horizontalgap
    layout_height = layout.row_count * y_step - layout.verticalgap

    fig_width_in = layout_width + 2 * layout.padding
    fig_height_in = layout_height + 2 * layout.padding

    if ax is None:
        figure, axes = plt.subplots(figsize=(fig_width_in, fig_height_in))
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        axes.set_position((0, 0, 1, 1))  # force axes to fill the whole figure
    else:
        figure, axes = ax.get_figure(root=True), ax
        assert isinstance(figure, Figure)
    axes.set_aspect("equal")
    axes.axis("off")

    text_outline_effects = [
        patheffects.Stroke(linewidth=sign_options.fontoutlinewidth, foreground="black"),
        patheffects.Normal(),
    ]

    items = _normalize_colors_and_adjust_item_order(
        colors=colors,
        labels=labels,
        reverse_display_order=reverse_display_order,
    )

    for index, (face_color, label_text) in enumerate(items):
        _draw_sign(
            axes=axes,
            index=index,
            face_color=face_color,
            label_text=label_text,
            sign_options=sign_options,
            orientation=orientation,
            item_count=layout.item_count,
            row_count=layout.row_count,
            column_count=layout.column_count,
            x_step=x_step,
            y_step=y_step,
            radius=layout.radius,
            text_outline_effects=text_outline_effects,
        )

    axes.set_xlim(-layout.padding, layout_width + layout.padding)
    axes.set_ylim(-layout.padding, layout_height + layout.padding)

    if filepath is not None:
        figure.savefig(filepath, bbox_inches="tight", pad_inches=0)

    return figure, axes
