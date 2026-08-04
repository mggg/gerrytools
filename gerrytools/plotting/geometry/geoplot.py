import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting._figure_io import save_figure
from gerrytools.plotting.geometry._labels import (
    LabelOptions,
    _merge_label_style_arg,
    _queue_label_request,
)
from gerrytools.plotting.geometry._layers import (
    ColormapLayer,
    _CategoricalColorLayer,
    _ContinuousColorLayer,
    _Layer,
)
from gerrytools.plotting.geometry._layers._marker import _normalize_label_key
from gerrytools.plotting.geometry.geoplotbase import GeoPlotBase
from gerrytools.plotting.mpl.geoplot_options import (
    ColorbarOptions,
    _ColorbarLayoutOptions,
)
from gerrytools.plotting.mpl.label_text_options import LabelStyle
from gerrytools.typing import (
    Color,
    CRSLike,
    GeoColorMap,
    MplCompatibleColor,
)


def _colorbar_rect(
    *,
    region: tuple[float, float, float, float],
    fig_size_inches: tuple[float, float],
    shrink: float | None,
    aspect: float | None,
    orientation: str = "vertical",
) -> tuple[float, float, float, float]:
    """Compute a colorbar axes rectangle in figure coordinates.

    matplotlib ignores shrink/aspect when a ``cax`` is supplied, so the rect is sized from
    them directly: ``shrink`` scales the bar along its long axis within ``region`` (and
    centers it there), and ``aspect`` is the bar's physical height-to-width ratio. A None
    ``aspect`` keeps the region's short-axis size.

    Args:
        region (tuple[float, float, float, float]): Bounding region
            ``(x0, y0, width, height)`` in figure coordinates.
        fig_size_inches (tuple[float, float]): Figure ``(width, height)`` in inches.
        shrink (float | None): Long-axis scale factor; None means 1.0.
        aspect (float | None): Physical height-to-width ratio, or None.
        orientation (str, optional): ``"vertical"`` or ``"horizontal"``. Defaults to
            ``"vertical"``.

    Returns:
        tuple[float, float, float, float]: The ``(left, bottom, width, height)`` rect.
    """
    x0, y0, region_width, region_height = region
    fig_width, fig_height = fig_size_inches
    shrink_value = 1.0 if shrink is None else float(shrink)

    if orientation == "vertical":
        bar_height = region_height * shrink_value
        bar_bottom = y0 + (region_height - bar_height) / 2
        bar_width = region_width
        if aspect is not None:
            bar_width = (bar_height * fig_height / float(aspect)) / fig_width
        return (float(x0), float(bar_bottom), float(bar_width), float(bar_height))

    bar_width = region_width * shrink_value
    bar_left = x0 + (region_width - bar_width) / 2
    bar_height = region_height
    if aspect is not None:
        bar_height = (bar_width * fig_width / float(aspect)) / fig_height
    return (float(bar_left), float(y0), float(bar_width), float(bar_height))


@dataclass(frozen=True, slots=True)
class _ColorbarRequest:
    layer: ColormapLayer
    label: str | None = None  # override label shown on the bar
    zorder: int = 0  # used only for ordering colorbars
    options: ColorbarOptions | None = None  # optional per-bar overrides (optional feature)


class GeoPlot(GeoPlotBase):
    """A class for creating geographic plots with multiple colored layers.

    Attributes:
        gdf (GeoDataFrame): The base GeoDataFrame for the plot.
        fig (Figure): The Matplotlib Figure object.
        target_crs (CRSLike | None): The target CRS for reprojecting geometries.
        silent (bool): Whether to suppress informational output throughout the rendering process.
    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        *,
        dpi: int | None = None,
        title: str | None = None,
        show_axis: bool = False,
        target_crs: CRSLike | None = None,
        default_outline: bool = True,
        silent: bool = True,
    ) -> None:
        """Initialize a GeoPlot.

        Args:
            gdf (GeoDataFrame): The base GeoDataFrame for the plot.
            dpi (int | None): The DPI for the Matplotlib Figure. Defaults to 300.
            title (str | None): The title of the plot. Defaults to None.
            show_axis (bool): Whether to show axis ticks and labels. Default is False.
            target_crs (CRSLike | None): The target CRS for reprojecting geometries. If None, uses
                the CRS of `gdf`. Default is None.
            default_outline (bool): Whether to include a default outline layer around the geometries
                in `gdf`. Default is True.
            silent (bool): Whether to suppress informational output throughout the rendering
                process. Default is True.
        """
        super().__init__(
            gdf,
            dpi=dpi,
            title=title,
            show_axis=show_axis,
            target_crs=target_crs,
            default_outline=default_outline,
            silent=silent,
        )

        self._colorbar_layout_options: _ColorbarLayoutOptions = _ColorbarLayoutOptions()
        self._colorbar_requests: list[_ColorbarRequest] = []

        self._choropleth_layers: list[_ContinuousColorLayer] = []
        self._districting_plan_layers: list[_CategoricalColorLayer] = []

        self._colorbar_axes: list[Axes] = []
        self._colorbar_histories: dict[Axes, list[Axes]] = {}

    @deferred_axis_update
    def add_choropleth_layer(
        self,
        column: str,
        *,
        geo_source: GeoDataFrame | None = None,
        colormap: str | Colormap = "Purples",
        missing_color: MplCompatibleColor | None = "lightgrey",
        facealpha: float | None = None,
        edgecolor: Color | None = "none",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        vmin: float | None = None,
        vmax: float | None = None,
        bins: int | list[float] | None = None,
        show_colorbar: bool = False,
        colorbar_label: str | None = None,
        colorbar_options: ColorbarOptions | None = None,
        zorder: int = 0,
    ) -> ColormapLayer:
        """Add a choropleth layer to the GeoPlotBase.

        Args:
            column (str): The data column to use for color mapping.
            geo_source (GeoDataFrame | None): The GeoDataFrame source for the layer. If None, uses
                the base gdf of the GeoPlotBase. Default is None.
            colormap (str | Colormap): The colormap to use for color mapping. Default is "Purples".
            missing_color (MplCompatibleColor | None): Color to use for missing data. Default is
                "lightgrey".
            facealpha (float | None): Alpha transparency for face colors. Default is None.
            edgecolor (Color | None): Color for geometry edges. Pass ``None`` for no
                edge. Default is "none".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
            vmin (float | None): Lower bound for color mapping range. Default is None which then
                uses the minimum value in the data.
            vmax (float | None): Upper bound for color mapping range. Default is None which then
                uses the maximum value in the data.
            show_colorbar (bool): Whether to show a colorbar for this layer. Default is False.
            colorbar_label (str | None): Label for the colorbar. Default is None which then uses the
                column name.
            colorbar_options (ColorbarOptions | None): Options for customizing the colorbar. Default
                is None.
            bins (int | list[float] | None): Optional binning specification for discrete intervals.
                Default is None.
            zorder (int): Z-order for rendering. Default is 0.

        Returns:
            ColormapLayer: The layer object, which can be passed to ``save_colorbar()``.
        """
        if geo_source is None:
            geo_source = self.gdf
        layer = _ContinuousColorLayer(
            geometry_source=geo_source,
            column=column,
            colormap=colormap,
            missing_color=missing_color,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            vmin=vmin,
            vmax=vmax,
            bins=bins,
            zorder=zorder,
        )
        self._choropleth_layers.append(layer)

        if show_colorbar:
            self._colorbar_requests.append(
                _ColorbarRequest(
                    layer=layer,
                    label=colorbar_label,
                    zorder=zorder,
                    options=colorbar_options,
                )
            )

        return layer

    @deferred_axis_update
    def add_colorbar(
        self,
        layer: ColormapLayer,
        *,
        label: str | None = None,
        options: ColorbarOptions | None = None,
        zorder: int = 0,
    ) -> None:
        """Attach a colorbar to an already-added choropleth layer.

        The post-hoc counterpart of ``add_choropleth_layer(..., show_colorbar=True)``: pass
        the handle that ``add_choropleth_layer()`` returned. The next build renders the
        colorbar alongside the map.

        Args:
            layer (ColormapLayer): The layer handle returned by ``add_choropleth_layer()``.
            label (str | None): Label for the colorbar. Defaults to None, which uses the
                layer's column name.
            options (ColorbarOptions | None): Options for customizing the colorbar.
                Defaults to None.
            zorder (int): Z-order used to sort multiple colorbars. Defaults to 0.
        """
        self._colorbar_requests.append(
            _ColorbarRequest(
                layer=layer,
                label=label,
                zorder=zorder,
                options=options,
            )
        )

    @deferred_axis_update
    def add_districting_plan_layer(
        self,
        plan_column: str,
        *,
        geo_source: GeoDataFrame | None = None,
        dissolve: bool = False,
        show_labels: bool = False,
        label_style: LabelStyle | str | None = None,
        label_options: LabelOptions | None = None,
        colormap: GeoColorMap | None = "districtr",
        missing_color: MplCompatibleColor | None = "lightgrey",
        facealpha: float | None = None,
        edgecolor: Color | None = "none",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        zorder: int = 2,
    ) -> None:
        """Add a districting plan layer to the GeoPlotBase.

        Args:
            plan_column (str): The column containing district identifiers.
            geo_source (GeoDataFrame | None): The GeoDataFrame source for the layer. If None, uses
                the base gdf of the GeoPlotBase. Default is None.
            dissolve (bool): Whether to dissolve geometries by district. Default is False.
            show_labels (bool): Whether to show district labels. Default is False.
            label_style (LabelStyle | str | None): Shorthand for
                ``label_options.label_style``: a
                ``LabelStyle`` or registered style name (e.g. ``"badge"``, ``"halo"``).
                Mutually exclusive with a ``label_options`` that carries its own label style.
                Defaults to None.
            label_options (LabelOptions | None): Bundled label styling and placement options,
                e.g. ``LabelOptions(label_style="halo")`` for the districtr-style numbers or
                ``"badge"`` for wheat circles, plus per-label adjustments, font sizes, and
                excluded district labels. Default is None.
            colormap (GeoColorMap | None): Color mapping specification. Can be a single color, a
                named colormap, a Colormap object, or a mapping from district identifiers to colors.
                A string is resolved as a registered Matplotlib colormap name first and as a flat
                color otherwise; a string that is both (e.g. ``"pink"``) resolves as the colormap
                and emits a ``UserWarning``. Pass a ``Colormap`` instance or an RGBA/hex value to
                force one meaning. Default is "districtr".
            missing_color (MplCompatibleColor | None): Color to use for missing data. Default is
                "lightgrey".
            facealpha (float | None): Alpha transparency for face colors. Default is None.
            edgecolor (Color | None): Color for geometry edges. Pass ``None`` for no
                edge. Default is "none".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
            zorder (int): Z-order for rendering. Default is 2.
        """
        if geo_source is None:
            plan_gdf = self.gdf
        else:
            plan_gdf = geo_source

        if dissolve:
            plan_gdf = GeoDataFrame(plan_gdf.dissolve(by=plan_column).reset_index())

        layer = _CategoricalColorLayer(
            geometry_source=plan_gdf,
            column=plan_column,
            colormap=colormap,
            missing_color=missing_color,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            zorder=zorder,
        )
        self._districting_plan_layers.append(layer)

        if show_labels:
            dissolved_plan_gdf = (
                plan_gdf
                if dissolve
                else GeoDataFrame(plan_gdf.dissolve(by=plan_column).reset_index())
            )
            _queue_label_request(
                self._label_requests,
                gdf=dissolved_plan_gdf,
                label_column=plan_column,
                options=_merge_label_style_arg(label_style, label_options),
                # District labels display in normalized form: "01" and 1 both render as "1".
                label_format_fn=_normalize_label_key,
                zorder=zorder + 1,
                dissolved=True,
            )

    @deferred_axis_update
    def set_colorbar_layout(
        self,
        *,
        outer_pad: float | None = None,
        inner_pad: float | None = None,
        width: float | None = None,
        right_margin: float | None = None,
    ) -> None:
        """Set the spacing between colorbars in GeoPlotBase.

        All arguments are optional; only those provided will be updated.

        Args:
            outer_pad (float | None): Padding between the colorbar and the plot edges (figure-
                relative). Default is None.
            inner_pad (float | None): Padding between the colorbar and the main plot area (figure-
                relative). Default is None.
            width (float | None): Width of the colorbar (figure-relative). Default is None.
            right_margin (float | None): Margin to the right of the colorbar (figure-relative).
                Default is None.
        """
        cb_options = self._colorbar_layout_options

        if outer_pad is not None:
            cb_options.outer_pad = float(outer_pad)
        if inner_pad is not None:
            cb_options.inner_pad = float(inner_pad)
        if width is not None:
            cb_options.width = float(width)
        if right_margin is not None:
            cb_options.right_margin = float(right_margin)

    def _layer_groups(self) -> list[tuple[str, Sequence[_Layer]]]:
        """Choropleth and districting-plan groups draw beneath the base groups."""
        return [
            ("choropleth", self._choropleth_layers),
            ("districting plan", self._districting_plan_layers),
            *super()._layer_groups(),
        ]

    def bind_to_ax(self, ax: Axes | None) -> None:
        """Retarget the plot without retaining colorbars from the previous figure.

        Binding to an axes immediately renders the accumulated layers and styles there. Prior
        output on an external axes remains unchanged. Pass ``None`` to create a fresh figure and
        defer rendering until :attr:`ax`, :meth:`show`, or :meth:`save` is accessed.

        Args:
            ax (matplotlib.axes.Axes | None): The matplotlib axes to render onto, or ``None`` to
                return to lazy rendering on a fresh figure.
        """
        if ax is self._ax:
            self._clear_colorbars_and_reset_layout()
        else:
            # Rebind is non-destructive. Keep inactive colorbars so revisiting their axes can
            # replace them instead of adding another copy.
            if self._colorbar_axes and (
                self._figure_is_shared or (ax is not None and ax.figure is self.fig)
            ):
                self._colorbar_histories[self._ax] = self._colorbar_axes
            self._colorbar_axes = self._colorbar_histories.pop(ax, []) if ax is not None else []
        super().bind_to_ax(ax)
        if ax is None:
            self._colorbar_histories.pop(self._ax, None)

    def _clear_colorbars_and_reset_layout(self) -> None:
        """Clear any existing colorbars and reset layout to default.

        ``subplots_adjust`` is only safe to call on a figure gerrytools owns. When the plot is bound
        to caller-owned axes, we share that figure and must not mutate its global layout.
        """
        for cax in list(self._colorbar_axes):
            if cax in self.fig.axes:
                cax.remove()
        self._colorbar_axes = []

        # reset layout so we don't keep a shrunken main axes
        if not self._figure_is_shared:
            self.fig.subplots_adjust(right=0.98, bottom=0.11)
            self.fig.canvas.draw_idle()

    @staticmethod
    def _style_colorbar(
        colorbar,
        cb_ax: Axes,
        cb_options: ColorbarOptions,
        label_text: str | None,
        layer_defaults: dict,
    ) -> None:
        """Apply shared tick, label, and override styling to a colorbar.

        Used by both the in-figure colorbars and ``save_colorbar()`` so the two render
        identically for the same options.

        Args:
            colorbar (Colorbar): The Matplotlib colorbar to style.
            cb_ax (Axes): The axes the colorbar occupies.
            cb_options (ColorbarOptions): The styling options to apply.
            label_text (str | None): The label to draw, or None for no label.
            layer_defaults (dict): Layer-supplied defaults, e.g. bin-edge ticks.
        """
        layer_ticks = layer_defaults.get("ticks")
        if isinstance(layer_ticks, list):
            colorbar.set_ticks(layer_ticks)

        if label_text is not None:
            # Pass only the options that were set: an explicit None (e.g. rotation=None)
            # would reset matplotlib's orientation-dependent default instead of keeping it.
            label_kwargs: dict = {}
            if cb_options.label_fontsize is not None:
                label_kwargs["fontsize"] = cb_options.label_fontsize
            if cb_options.label_rotation is not None:
                label_kwargs["rotation"] = cb_options.label_rotation
            if cb_options.label_pad is not None:
                label_kwargs["labelpad"] = cb_options.label_pad
            colorbar.set_label(str(label_text), **label_kwargs)

        cb_ax.tick_params(labelsize=cb_options.tick_fontsize, pad=cb_options.tick_pad)

        if cb_options.force_ticks is not None:
            colorbar.set_ticks(cb_options.force_ticks)
        if cb_options.force_ticklabels is not None:
            colorbar.set_ticklabels(cb_options.force_ticklabels)

        if cb_options.max_n_ticks is not None and cb_options.force_ticks is None:
            ticks = list(colorbar.get_ticks())
            if len(ticks) > cb_options.max_n_ticks:
                # ceil so the kept tick count never exceeds the requested maximum.
                step = math.ceil(len(ticks) / cb_options.max_n_ticks)
                colorbar.set_ticks(ticks[::step])

    @staticmethod
    def _make_colorbar(
        fig: Figure, cax: Axes, mappable: ScalarMappable, options: ColorbarOptions
    ) -> Colorbar:
        """Create a colorbar on ``cax`` from shared ``ColorbarOptions``.

        The single construction point for in-figure colorbars and ``save_colorbar()``, so
        the two render identically for the same options.
        """
        return fig.colorbar(
            mappable=mappable,
            cax=cax,
            orientation=options.orientation,
            extend=options.extend,
            format=options.format,
        )

    def _draw_colorbars(self) -> None:
        """Draw colorbars for all requested layers.

        Vertical bars occupy a right-hand strip and stack rightward; horizontal bars occupy
        a strip below the map axes and stack downward.
        """
        self._clear_colorbars_and_reset_layout()

        if not self._colorbar_requests:
            return

        # Sort requests by their requested zorder (usually layer.zorder)
        sorted_cb_requests = sorted(self._colorbar_requests, key=lambda r: int(r.zorder))
        resolved_requests = [
            (request, request.options if request.options is not None else ColorbarOptions())
            for request in sorted_cb_requests
        ]
        n_horizontal = sum(
            1 for _, cb_options in resolved_requests if cb_options.orientation == "horizontal"
        )
        n_vertical = len(resolved_requests) - n_horizontal

        cb_global_options = self._colorbar_layout_options
        outer_pad = float(cb_global_options.outer_pad)
        inner_pad = float(cb_global_options.inner_pad)
        width = float(cb_global_options.width)
        right_margin = float(cb_global_options.right_margin)

        def strip_size(n_bars: int) -> float:
            return n_bars * width + (n_bars - 1) * inner_pad + outer_pad + right_margin

        if not self._figure_is_shared:
            if n_vertical:
                self.fig.subplots_adjust(right=max(0.05, 1.0 - strip_size(n_vertical)))
            if n_horizontal:
                self.fig.subplots_adjust(bottom=min(0.95, strip_size(n_horizontal)))
            self.fig.canvas.draw_idle()

        main_pos = self._ax.get_position()
        x0 = float(main_pos.x1 + outer_pad)
        y0 = float(main_pos.y0)
        h = float(main_pos.height)
        horizontal_top = float(main_pos.y0 - outer_pad)

        vertical_index = 0
        horizontal_index = 0
        for cb_request, cb_options in resolved_requests:
            layer = cb_request.layer

            if cb_options.orientation == "horizontal":
                yi = horizontal_top - horizontal_index * (width + inner_pad) - width
                region = (float(main_pos.x0), yi, float(main_pos.width), width)
                horizontal_index += 1
            else:
                xi = x0 + vertical_index * (width + inner_pad)
                region = (xi, y0, width, h)
                vertical_index += 1

            fig_w, fig_h = self.fig.get_size_inches()
            rect = _colorbar_rect(
                region=region,
                fig_size_inches=(float(fig_w), float(fig_h)),
                shrink=cb_options.shrink,
                aspect=cb_options.aspect,
                orientation=cb_options.orientation,
            )
            cb_ax = self.fig.add_axes(rect)
            self._colorbar_axes.append(cb_ax)

            mappable, layer_defaults = layer.mappable()
            colorbar = self._make_colorbar(self.fig, cb_ax, mappable, cb_options)

            # label: request override > column > none
            label_text = cb_request.label if cb_request.label is not None else layer.column
            self._style_colorbar(colorbar, cb_ax, cb_options, label_text, layer_defaults)

    def _build_plot(self) -> None:
        """Render all layers, then the colorbars, tracking every artist created."""
        super()._build_plot()
        self._draw_colorbars()

    def save_colorbar(
        self,
        layer: ColormapLayer,
        filepath: str,
        *,
        figsize: tuple[float, float] = (1.0, 4.0),
        orientation: str | None = None,
        options: ColorbarOptions | None = None,
        label: str | None = None,
        ax_rect: tuple[float, float, float, float] | None = None,
        **kwargs: object,
    ) -> None:
        """Save a standalone colorbar image for a choropleth layer.

        The bar is styled exactly like an in-figure colorbar with the same options, so a
        separately printed key matches its map panels. The image is cropped tight with no
        padding by default; pass ``bbox_inches`` / ``pad_inches`` to override.

        Args:
            layer (ColormapLayer): The layer whose colorbar to save. This is the value
                returned by ``add_choropleth_layer()``.
            filepath (str): Output file path.
            figsize (tuple[float, float]): Figure size (width, height) in inches.
                Defaults to (1.0, 4.0) for a vertical bar.
            orientation (str | None): ``"vertical"`` or ``"horizontal"``. Defaults to
                None, which uses ``options.orientation``.
            options (ColorbarOptions | None): Tick, label, and sizing options; ``shrink``
                scales the bar within the figure and ``aspect`` sets its physical
                height-to-width ratio. Defaults to None.
            label (str | None): Label override. Defaults to None, which uses the layer's
                column name; pass ``""`` for no label.
            ax_rect (tuple[float, float, float, float] | None): Explicit colorbar axes
                rectangle in figure coordinates ``(left, bottom, width, height)``,
                overriding the shrink/aspect sizing. Defaults to None.
            **kwargs (object): Additional keyword arguments passed to ``Figure.savefig``.

        Raises:
            ValueError: If ``orientation`` is not ``"vertical"`` or ``"horizontal"``.
        """
        cb_options = options if options is not None else ColorbarOptions()
        orientation = orientation if orientation is not None else cb_options.orientation
        if orientation == "vertical" or orientation == "horizontal":
            cb_options = replace(cb_options, orientation=orientation)
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'.")

        mappable, layer_defaults = layer.mappable()

        # Guard every operation after creation so failures cannot leak the figure.
        cb_fig = plt.figure(figsize=figsize, dpi=self.fig.dpi)
        try:
            if ax_rect is None:
                fig_w, fig_h = cb_fig.get_size_inches()
                ax_rect = _colorbar_rect(
                    region=(0.0, 0.0, 1.0, 1.0),
                    fig_size_inches=(float(fig_w), float(fig_h)),
                    shrink=cb_options.shrink,
                    aspect=20.0 if cb_options.aspect is None else float(cb_options.aspect),
                    orientation=orientation,
                )
            cb_ax = cb_fig.add_axes(ax_rect)

            colorbar = self._make_colorbar(cb_fig, cb_ax, mappable, cb_options)

            label_text = label if label is not None else layer.column
            self._style_colorbar(colorbar, cb_ax, cb_options, label_text, layer_defaults)

            kwargs.setdefault("bbox_inches", "tight")
            kwargs.setdefault("pad_inches", 0)
            save_figure(cb_fig, filepath, **kwargs)
        finally:
            plt.close(cb_fig)
