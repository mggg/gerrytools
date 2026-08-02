from collections.abc import Sequence
from dataclasses import replace
from typing import Literal, cast
from warnings import warn

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point

from gerrytools.plotting._artist_registry import _ArtistRegistry
from gerrytools.plotting._axes_backed import _AxesBackedPlot, deferred_axis_update
from gerrytools.plotting._axes_state import Unit, _ManagedAxesState, _recompute_data_limits
from gerrytools.plotting.data._axis_api import _TitleApiMixin, _TitleText
from gerrytools.plotting.geometry._labels import (
    _DEFAULT_LABEL_FONT,
    _INVISIBLE_MARKER,
    LabelOptions,
    _draw_deferred_labels,
    _label_keep_mask,
    _LabelRequest,
    _merge_label_style_arg,
    _queue_label_request,
)
from gerrytools.plotting.geometry._layers import (
    _as_geoseries,
    _CategoricalColorLayer,
    _Layer,
    _MarkerLayer,
)
from gerrytools.plotting.geometry._layers._base import _mask_geoseries, _to_target_crs
from gerrytools.plotting.mpl.label_text_options import (
    LabelBoxOptions,
    LabelFontOptions,
    LabelStyle,
)
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.typing import (
    Color,
    CRSLike,
    GeoSource,
)


def _resolve_points(
    points_geoseries: gpd.GeoSeries | None,
    latlon_list: Sequence[tuple[float, float]] | None,
    *,
    input_crs: CRSLike | None,
    plot_crs: CRSLike | None,
) -> gpd.GeoSeries:
    """Resolve marker/label point input to a ``GeoSeries``, enforcing exactly one source.

    ``input_crs`` always means the CRS of the caller's coordinates. The returned series is
    tagged with that source CRS; layers reproject to the plot CRS at render time.

    Args:
        points_geoseries (gpd.GeoSeries | None): Point geometries, or None when using
            ``latlon_list``.
        latlon_list (Sequence[tuple[float, float]] | None): (latitude, longitude) pairs, or None
            when using ``points_geoseries``.
        input_crs (CRSLike | None): CRS of the caller's coordinates. For ``latlon_list`` it
            defaults to EPSG:4326 (lat/lon); for a CRS-less ``points_geoseries`` it tags the
            series. A ``points_geoseries`` that already carries a CRS keeps it.
        plot_crs (CRSLike | None): The CRS the plot renders in. Used only to reject
            ``latlon_list`` input when the plot has no CRS to reproject into.

    Returns:
        gpd.GeoSeries: The resolved point geometries, tagged with their source CRS.

    Raises:
        ValueError: If neither or both point sources are given, or if ``latlon_list`` points
            are used on a plot with no CRS.
    """
    if points_geoseries is None and latlon_list is None:
        raise ValueError("Either `points_geoseries` or `latlon_list` must be set.")
    if points_geoseries is not None and latlon_list is not None:
        raise ValueError(
            "Only one of `points_geoseries` or `latlon_list` may be set at a time.",
        )

    if latlon_list is not None:
        if plot_crs is None:
            raise ValueError(
                "latlon_list points cannot be placed on a plot with no CRS: use a base gdf "
                "with a CRS (or pass points_geoseries in map coordinates).",
            )
        # The pairs are (latitude, longitude), i.e. (y, x); Point takes (x, y).
        point_geometries = gpd.GeoSeries(
            [Point(float(longitude), float(latitude)) for latitude, longitude in latlon_list],
            crs=input_crs if input_crs is not None else "EPSG:4326",
        )
        return point_geometries

    point_geometries = cast(gpd.GeoSeries, points_geoseries)
    if getattr(point_geometries, "crs", None) is None and input_crs is not None:
        point_geometries = point_geometries.set_crs(input_crs)
    return point_geometries


class GeoPlotBase(_TitleApiMixin, _AxesBackedPlot):
    """A class for creating geographic plots with multiple layers.

    Attributes:
        gdf (GeoDataFrame): The base GeoDataFrame for the plot.
        fig (Figure): The Matplotlib Figure object.
        target_crs (CRSLike | None): The target CRS for reprojecting geometries.
        silent (bool): Whether to suppress informational output throughout the rendering process.
    """

    _non_gui_filename = "geoplot.png"
    _non_gui_prefix = "GeoPlotBase"

    def __init__(
        self,
        gdf: GeoDataFrame,
        *,
        dpi: int | None = None,
        title: str | None = None,
        show_axis: bool = False,
        target_crs: CRSLike | None = None,
        default_outline: bool = True,
        silent: bool = False,
    ) -> None:
        """Initialize a GeoPlotBase.

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
                process. Default is False.
        """
        self.gdf = gdf

        # --- Pass 1: resolve self._ax + self.fig + _figure_is_shared --- The dpi is remembered so
        # ``bind_to_ax(None)`` can recreate a fresh figure with the same dpi as construction; the
        # figure size stays None because geometry plots size to their data.
        self._figure_size = None
        self._figure_dpi = dpi if dpi is not None else 300
        self._attach_axes(None)

        # --- Pass 2: backing fields (no opinion until step 4 reapplies args) ---
        self._title_text = _TitleText(unit="title")
        self._show_axis: bool = False
        self._target_crs: CRSLike | None = (
            target_crs if target_crs is not None else getattr(gdf, "crs", None)
        )

        self._xlim: tuple[float, float] | None = None
        self._ylim: tuple[float, float] | None = None

        self._outline_layers: list[_CategoricalColorLayer] = []
        self._highlight_layers: list[_CategoricalColorLayer] = []
        self._marker_layers: list[_MarkerLayer] = []

        self._label_requests: list[_LabelRequest] = []
        self._last_label_positions: dict[str, Point] | None = None
        self.silent = silent

        # --- Pass 3: artist registry + managed-axes state ---
        self._artists = _ArtistRegistry()
        self._axes_state = _ManagedAxesState()
        self._axes_state.initialize_from_ax(self._ax)

        # --- Pass 4: re-apply non-default constructor args via reclaim path ---
        if title is not None:
            self.title = title
        if show_axis:
            self.show_axis = show_axis

        if default_outline:
            fully_dissolved_geos = GeoSeries(gdf.geometry.union_all(), crs=gdf.crs)
            self.add_outline_layer(
                geo_source=fully_dissolved_geos,
                edgecolor="black",
                edgewidth=0.5,
            )

    @property
    def target_crs(self) -> CRSLike | None:
        """Coordinate reference system used to render geometry layers."""
        return self._target_crs

    @target_crs.setter
    @deferred_axis_update
    def target_crs(self, value: CRSLike | None) -> None:
        """Set the render CRS, discarding limits measured in the old projection."""
        self._set_target_crs(value)

    def _set_target_crs(self, value: CRSLike | None) -> None:
        """Apply a CRS change and release limits measured in the old projection."""
        if value != self._target_crs and (self._xlim is not None or self._ylim is not None):
            warn(
                "Discarding axis limits set in the previous CRS; call focus_axes() or "
                "set_xlim/set_ylim again to reframe in the new projection.",
                stacklevel=2,
            )
            self._xlim = None
            self._ylim = None
            self._ax.set_autoscalex_on(True)
            self._ax.set_autoscaley_on(True)
            self._axes_state.release("x_limits", tuple(float(v) for v in self._ax.get_xlim()))
            self._axes_state.release("y_limits", tuple(float(v) for v in self._ax.get_ylim()))
        self._target_crs = value

    @deferred_axis_update
    def add_outline_layer(
        self,
        *,
        geo_source: GeoDataFrame | GeoSeries | None = None,
        geometry_mask: pd.Series | None = None,
        dissolve_column: str | None = None,
        edgecolor: Color | None = "black",
        edgealpha: float | None = None,
        edgewidth: float = 0.5,
        show_labels: bool = False,
        label_style: LabelStyle | str | None = None,
        label_options: LabelOptions | None = None,
        zorder: int = 3,
    ) -> None:
        """Add an outline layer to the GeoPlotBase.

        Args:
            geo_source (GeoDataFrame | GeoSeries | None): The GeoDataFrame or GeoSeries source for
                the layer. If None, uses the base gdf of the GeoPlotBase. Default is None.
            geometry_mask (pd.Series | None): Optional boolean mask aligned to the input
                ``geo_source`` rows. With ``dissolve_column`` set, the mask filters input rows
                before dissolving. Default is None.
            dissolve_column (str | None): Optional column to dissolve geometries by before
                outlining. Default is None.
            edgecolor (Color | None): Color for geometry edges. Pass ``None`` for no
                edge. Default is "black".
            edgealpha (float | None): Alpha transparency for edge colors. Default is None.
            edgewidth (float): Width of geometry edges. Default is 0.5.
            show_labels (bool): Whether to show labels on the outlined geometries. Default is False.
            label_style (LabelStyle | str | None): Shorthand for
                ``label_options.label_style``: a
                ``LabelStyle`` or registered style name (e.g. ``"badge"``, ``"halo"``).
                Mutually exclusive with a ``label_options`` that carries its own label style.
                Defaults to None.
            label_options (LabelOptions | None): Bundled label styling and placement options
                (label style or font/box options, per-label adjustments and font sizes, and
                excluded labels). When None (or with a None ``font_options`` and no label style),
                labels use the default geography font: fontcolor="black", fontsize=4,
                fontweight="roman", outlinecolor="grey", outlinewidth=0.2. Default is None.
            zorder (int): Z-order for rendering. Default is 3.

        Raises:
            TypeError: If dissolving is requested for a source that is not a GeoDataFrame.
            ValueError: If labels are requested without a dissolve column or options are invalid.
        """
        if geo_source is None:
            geo_source = self.gdf

        # Validate before touching layer state, so a failed call registers nothing.
        if show_labels and dissolve_column is None:
            raise ValueError(
                "'dissolve_column' must be set to add labels to an outline layer",
            )

        dissolved: GeoDataFrame | None = None
        if dissolve_column is not None:
            if not isinstance(geo_source, GeoDataFrame):
                raise TypeError(
                    "Tried to dissolve geo_source of type "
                    f"{type(geo_source).__name__!r}; geo_source must be a GeoDataFrame",
                )
            # The mask is aligned to the input rows, so filter before dissolving; the layer
            # then gets the dissolved frame with no further mask.
            if geometry_mask is not None:
                geo_source = GeoDataFrame(geo_source.iloc[geometry_mask.to_numpy(dtype=bool)])
                geometry_mask = None
            dissolved = GeoDataFrame(geo_source.dissolve(by=dissolve_column).reset_index())
            geo_source = dissolved

        layer = _CategoricalColorLayer(
            geometry_source=geo_source,
            geometry_mask=geometry_mask,
            colormap="none",
            missing_color="none",
            facealpha=0.0,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            zorder=zorder,
        )
        self._outline_layers.append(layer)

        if show_labels and dissolved is not None and dissolve_column is not None:
            _queue_label_request(
                self._label_requests,
                gdf=dissolved,
                label_column=dissolve_column,
                options=_merge_label_style_arg(label_style, label_options),
                zorder=zorder + 1,
                dissolved=True,
            )

    @deferred_axis_update
    def add_highlight_layer(
        self,
        label_column: str | None = None,
        *,
        geo_source: GeoDataFrame | GeoSeries | None = None,
        geometry_mask: pd.Series | None = None,
        facecolor: Color | None = "gray",
        facealpha: float | None = 0.5,
        show_labels: bool = False,
        label_style: LabelStyle | str | None = None,
        label_options: LabelOptions | None = None,
        zorder: int = 10,
    ) -> None:
        """Add a highlight layer to the GeoPlotBase.

        Args:
            label_column (str | None): Optional column to label geometries by before highlighting.
                Default is None.
            geo_source (GeoDataFrame | GeoSeries | None): The GeoDataFrame or GeoSeries source for
                the layer. If None, uses the base gdf of the GeoPlotBase. Default is None.
            geometry_mask (pd.Series | None): Optional boolean mask to filter geometries. Default is
                None.
            facecolor (Color | None): Color for geometry faces. Pass ``None`` for no
                fill. Default is "gray".
            facealpha (float | None): Alpha transparency for face colors. Default is 0.5.
            show_labels (bool): Whether to show labels on the highlighted geometries. Default is
                False.
            label_style (LabelStyle | str | None): Shorthand for
                ``label_options.label_style``: a
                ``LabelStyle`` or registered style name (e.g. ``"badge"``, ``"halo"``).
                Mutually exclusive with a ``label_options`` that carries its own label style.
                Defaults to None.
            label_options (LabelOptions | None): Bundled label styling and placement options
                (label style or font/box options, per-label adjustments and font sizes, and
                excluded labels). When None (or with a None ``font_options`` and no label style),
                labels use the default geography font. Default is None.
            zorder (int): Z-order for rendering. Default is 10.

        Raises:
            TypeError: If labels are requested from a source that is not a GeoDataFrame.
            ValueError: If label inputs or styling options are invalid.
        """
        # Validate before touching layer state, so a failed call registers nothing.
        label_source: GeoDataFrame | None = None
        if show_labels:
            if label_column is None:
                raise ValueError(
                    "add_highlight_layer(show_labels=True) requires label_column=... to know "
                    "what to label. Example: dissolve_column='COUNTYFP10'."
                )
            if geo_source is None:
                raise ValueError(
                    "add_highlight_layer(show_labels=True) requires geo_source=... (a GeoDataFrame) "
                    "so the dissolve_column exists."
                )
            if not isinstance(geo_source, GeoDataFrame):
                raise TypeError(
                    "add_highlight_layer(show_labels=True) requires geo_source to be a GeoDataFrame "
                    f"so it has the label_column {label_column!r}. "
                    f"You passed {type(geo_source).__name__!r}. "
                    "Either pass a GeoDataFrame, or set show_labels=False."
                )
            label_source = geo_source

        if geo_source is None:
            geometries = self.gdf.geometry
        else:
            geometries = _as_geoseries(geo_source)

        if geometry_mask is not None:
            geometries = _mask_geoseries(geometries, geometry_mask)

        geometries = GeoSeries(geometries.union_all(), crs=geometries.crs)

        layer = _CategoricalColorLayer(
            geometry_source=geometries,
            colormap=facecolor,
            missing_color="none",
            facealpha=facealpha,
            edgecolor="none",
            edgealpha=None,
            edgewidth=0.0,
            zorder=zorder,
        )
        self._highlight_layers.append(layer)

        if label_source is not None and label_column is not None:
            if geometry_mask is not None:
                label_source = GeoDataFrame(label_source.iloc[geometry_mask.to_numpy(dtype=bool)])
            _queue_label_request(
                self._label_requests,
                gdf=label_source,
                label_column=label_column,
                options=_merge_label_style_arg(label_style, label_options),
                zorder=zorder + 1,
            )

    @deferred_axis_update
    def add_marker_layer(
        self,
        points_geoseries: gpd.GeoSeries | None = None,
        *,
        latlon_list: Sequence[tuple[float, float]] | None = None,
        input_crs: CRSLike | None = None,
        marker_options: PointMarkerOptions | None = None,
        show_labels: bool = True,
        labels: Sequence[str] | None = None,
        label_style: LabelStyle | str | None = None,
        label_options: LabelOptions | None = None,
        zorder: int = 2,
    ) -> None:
        """Add a layer of markers (points) to the GeoPlotBase.

        Args:
            points_geoseries (gpd.GeoSeries | None): A GeoSeries of Point geometries for the
                markers. If None, `latlon_list` must be provided. Default is None.
            latlon_list (Sequence[tuple[float, float]] | None): A sequence of (latitude, longitude)
                tuples for the marker locations. If None, `points_geoseries` must be provided.
                Default is None.
            input_crs (CRSLike | None, optional): The CRS of the input coordinates, for
                ``latlon_list`` or a CRS-less ``points_geoseries``. If None, ``latlon_list``
                is assumed EPSG:4326 (lat/lon) and a CRS-bearing ``points_geoseries`` keeps
                its own CRS. Points are reprojected to the plot CRS at render. Default is None.
            marker_options (PointMarkerOptions | None): Marker style settings.
                If None, uses the following defaults:
                    - markerfacecolor="white",
                    - markerfacealpha=1.0,
                    - marker="o",
                    - markersize=3.0,
                    - markeredgecolor="black",
                    - markeredgealpha=1.0,
                    - markeredgewidth=0.5.
                Default is None.
            show_labels (bool): Whether to show labels on the markers. Default is True.
            labels (Sequence[str] | None): Optional labels for each marker. Default is None.
            label_style (LabelStyle | str | None): Shorthand for
                ``label_options.label_style``: a
                ``LabelStyle`` or registered style name (e.g. ``"badge"``, ``"halo"``).
                Mutually exclusive with a ``label_options`` that carries its own label style.
                Defaults to None.
            label_options (LabelOptions | None): Bundled label styling and placement options
                (label style or font/box options, per-label adjustments and font sizes, and
                excluded labels). Styles may vary the box per label, e.g. equalizing badge
                circle diameters. When None (or with None ``font_options`` /
                ``box_options``), labels use default ``LabelFontOptions()`` and a disabled
                box. Default is None.
            zorder (int, optional): Z-order for rendering. Defaults to ``2``.

        Raises:
            ValueError: If point sources, labels, or styling options are invalid.
        """
        merged_options = _merge_label_style_arg(label_style, label_options)
        options = merged_options if merged_options is not None else LabelOptions()
        if marker_options is None:
            marker_options = PointMarkerOptions(
                markerfacecolor="white",
                markerfacealpha=1.0,
                marker="o",
                markersize=3.0,
                markeredgecolor="black",
                markeredgealpha=1.0,
                markeredgewidth=0.5,
            )
        label_font_options = (
            options.font_options if options.font_options is not None else LabelFontOptions()
        )
        label_box_options = (
            options.box_options
            if options.box_options is not None
            else LabelBoxOptions(enabled=False)
        )

        point_geometries = _resolve_points(
            points_geoseries,
            latlon_list,
            input_crs=input_crs,
            plot_crs=self.target_crs,
        )

        # Validate lengths before the exclude filter below: a mismatched ``labels`` would
        # otherwise surface as a pandas IndexError instead of this clear message.
        if labels is not None and len(labels) != len(point_geometries):
            raise ValueError("`labels` must have the same length as `point_geometries`.")

        # ``LabelOptions.exclude`` applies to every labeled layer: drop excluded labels and
        # their points here, matching the dissolved-path normalization semantics.
        if show_labels and labels is not None and options.exclude:
            keep = _label_keep_mask(labels, options.exclude)
            labels = [label for label, kept in zip(labels, keep) if kept]
            point_geometries = gpd.GeoSeries(point_geometries[keep])

        marker_layer = _MarkerLayer(
            point_geometries=point_geometries,
            labels=labels,
            marker_options=marker_options,
            show_labels=show_labels,
            label_style=options.resolved_label_style,
            label_adjustments=options.adjustments,
            label_fontsize=options.fontsize,
            label_font_options=label_font_options,
            label_box_options=label_box_options,
            zorder=zorder,
        )
        self._marker_layers.append(marker_layer)

    def add_label_layer(
        self,
        points_geoseries: gpd.GeoSeries | None = None,
        labels: Sequence[str] | None = None,
        *,
        latlon_list: Sequence[tuple[float, float]] | None = None,
        input_crs: CRSLike | None = None,
        label_style: LabelStyle | str | None = None,
        label_options: LabelOptions | None = None,
        zorder: int = 2,
    ) -> None:
        """Add a layer of text labels at point locations (no visible markers).

        Args:
            points_geoseries (gpd.GeoSeries | None): A GeoSeries of Point geometries for the
                labels. If None, `latlon_list` must be provided. Default is None.
            labels (Sequence[str] | None): Optional labels for each point. Default is None which
                results in numerical labels.
            latlon_list (Sequence[tuple[float, float]] | None): A sequence of (latitude, longitude)
                tuples for the label locations. If None, `points_geoseries` must be provided.
                Default is None.
            input_crs (CRSLike | None, optional): The CRS of the input coordinates, for
                ``latlon_list`` or a CRS-less ``points_geoseries``. If None, ``latlon_list``
                is assumed EPSG:4326 (lat/lon) and a CRS-bearing ``points_geoseries`` keeps
                its own CRS. Points are reprojected to the plot CRS at render. Default is None.
            label_style (LabelStyle | str | None): Shorthand for
                ``label_options.label_style``: a
                ``LabelStyle`` or registered style name (e.g. ``"badge"``, ``"halo"``).
                Mutually exclusive with a ``label_options`` that carries its own label style.
                Defaults to None.
            label_options (LabelOptions | None): Bundled label styling and placement options
                (label style or font/box options, per-label adjustments and font sizes, and
                excluded labels). When None (or with a None ``font_options`` and no label style),
                labels use the default geography font: fontcolor="black", fontsize=4,
                fontweight="roman", outlinecolor="grey", outlinewidth=0.2. Default is None.
            zorder (int, optional): Z-order for rendering. Defaults to ``2``.
        """
        merged_options = _merge_label_style_arg(label_style, label_options)
        options = merged_options if merged_options is not None else LabelOptions()
        point_geometries = _resolve_points(
            points_geoseries,
            latlon_list,
            input_crs=input_crs,
            plot_crs=self.target_crs,
        )

        if labels is None:
            labels = [str(i) for i in range(len(point_geometries))]

        if options.font_options is None and options.label_style is None:
            options = replace(options, font_options=_DEFAULT_LABEL_FONT)

        self.add_marker_layer(
            points_geoseries=point_geometries,
            marker_options=_INVISIBLE_MARKER,
            show_labels=True,
            labels=labels,
            label_options=options,
            zorder=zorder,
        )

    # ------------------------------------------------------------------
    # show_axis property — axis_visibility managed unit
    # ------------------------------------------------------------------

    @property
    def show_axis(self) -> bool:
        """Whether to render axis ticks/labels (False hides via ``ax.set_axis_off``)."""
        return self._show_axis

    @show_axis.setter
    @deferred_axis_update
    def show_axis(self, value: bool) -> None:
        self._show_axis = bool(value)
        if self._show_axis:
            self._ax.set_axis_on()
        else:
            self._ax.set_axis_off()
        self._axes_state.reclaim_and_mark("axis_visibility", bool(self._ax.axison))

    @deferred_axis_update
    def set_xlim(self, left: float, right: float) -> None:
        """Set x-axis limits to apply when the plot is built.

        Matches the matplotlib convention ``Axes.set_xlim(left, right)``.

        Args:
            left (float): The left x-axis limit.
            right (float): The right x-axis limit.
        """
        self._xlim = (float(left), float(right))
        self._ax.set_xlim(*self._xlim)
        self._axes_state.reclaim_and_mark("x_limits", tuple(float(v) for v in self._ax.get_xlim()))

    @deferred_axis_update
    def set_ylim(self, bottom: float, top: float) -> None:
        """Set y-axis limits to apply when the plot is built.

        Matches the matplotlib convention ``Axes.set_ylim(bottom, top)``.

        Args:
            bottom (float): The bottom y-axis limit.
            top (float): The top y-axis limit.
        """
        self._ylim = (float(bottom), float(top))
        self._ax.set_ylim(*self._ylim)
        self._axes_state.reclaim_and_mark("y_limits", tuple(float(v) for v in self._ax.get_ylim()))

    @deferred_axis_update
    def clear_limits(self) -> None:
        """Clear any stored x/y limits and return the axes to autoscaling.

        The next build recomputes limits from the drawn geometries; the managed limit
        units drop back to default ownership so external ``set_xlim``/``set_ylim`` calls
        are detected again afterwards.
        """
        self._xlim = None
        self._ylim = None
        self._ax.set_autoscalex_on(True)
        self._ax.set_autoscaley_on(True)
        self._recompute_data_limits()
        self._ax.autoscale_view()
        self._axes_state.release("x_limits", tuple(float(v) for v in self._ax.get_xlim()))
        self._axes_state.release("y_limits", tuple(float(v) for v in self._ax.get_ylim()))

    def _recompute_data_limits(self) -> None:
        """Recompute limits, adding collections that Matplotlib's ``relim`` skips."""
        _recompute_data_limits(self._ax)

    def focus_axes(
        self,
        *,
        geo_source: GeoSource | None = None,
        geometry_mask: pd.Series | None = None,
        pad: float | tuple[float, float] | tuple[float, float, float, float] = 0.02,
        pad_mode: Literal["fraction", "data"] = "fraction",
    ) -> None:
        """Set x/y limits to the (padded) bounding box of a geo_source.

        Args:
            geo_source (GeoSource | None, optional): GeoDataFrame or GeoSeries to focus on. Defaults
                to this plot's gdf. If None, will use the base gdf used to initialize GeoPlotBase.
                Defaults to None.
            geometry_mask (pd.Series | None): Optional boolean mask aligned to geo_source index. If
                None, will use all geometries in geosouce. Defaults to None.
            pad (float | tuple): Padding around bounds. A single float pads every side, a
                2-tuple ``(pad_x, pad_y)`` pads each axis symmetrically, and a 4-tuple
                ``(top, right, bottom, left)`` pads each side independently, so a panel
                can stay tight on two sides while leaving room for a key on the others.
                With ``pad_mode="fraction"`` values are fractions of the width/height
                (0.02 = 2%); with ``pad_mode="data"`` they are absolute data units.
                Defaults to 0.02.
            pad_mode (Literal): "fraction" or "data". Defaults to "fraction".

        Raises:
            ValueError: If no usable geometries remain or padding options are invalid.
        """
        if geo_source is None:
            geo_source = self.gdf

        geoseries = _as_geoseries(geo_source)

        if geometry_mask is not None:
            geoseries = _mask_geoseries(geoseries, geometry_mask)

        geoseries = _mask_geoseries(geoseries, geoseries.notna())
        geoseries = _mask_geoseries(geoseries, ~geoseries.is_empty)

        if geoseries.empty:
            raise ValueError(
                "focus_axes(): no geometries after applying mask / dropping empties. "
                "Double check your geometry_mask to make sure that it is a valid filter "
                "for the provided geo_source.",
            )

        geoseries = _to_target_crs(geoseries, self.target_crs)

        minx, miny, maxx, maxy = map(float, geoseries.total_bounds)

        width = maxx - minx
        height = maxy - miny

        if isinstance(pad, tuple) and len(pad) == 4:
            pad_top, pad_right, pad_bottom, pad_left = (float(value) for value in pad)
        elif isinstance(pad, tuple) and len(pad) == 2:
            pad_left = pad_right = float(pad[0])
            pad_top = pad_bottom = float(pad[1])
        elif isinstance(pad, tuple):
            raise ValueError("pad tuple must have 2 or 4 elements.")
        else:
            pad_top = pad_right = pad_bottom = pad_left = float(pad)

        if pad_mode == "fraction":
            # If width/height are 0 (single point/line), give a small default pad
            d_left = (width * pad_left) if width > 0 else pad_left
            d_right = (width * pad_right) if width > 0 else pad_right
            d_bottom = (height * pad_bottom) if height > 0 else pad_bottom
            d_top = (height * pad_top) if height > 0 else pad_top
        elif pad_mode == "data":
            d_left, d_right, d_bottom, d_top = pad_left, pad_right, pad_bottom, pad_top
        else:
            raise ValueError("pad_mode must be 'fraction' or 'data'.")

        self.set_xlim(minx - d_left, maxx + d_right)
        self.set_ylim(miny - d_bottom, maxy + d_top)

    # ------------------------------------------------------------------
    # The rebuild pipeline
    # ------------------------------------------------------------------

    def _layer_groups(self) -> list[tuple[str, Sequence[_Layer]]]:
        """Layer groups in draw order. Subclasses prepend their own groups."""
        return [
            ("marker", self._marker_layers),
            ("outline", self._outline_layers),
            ("highlight", self._highlight_layers),
        ]

    def _build_plot(self) -> None:
        """Render every layer group in order, tracking the artists each layer creates.

        Each layer's ``render()`` returns the matplotlib artists it created on the axes; we track
        them via ``self._artists`` so the next rebuild can remove gerrytools-managed artists without
        disturbing external content.
        """
        for group_name, layers in self._layer_groups():
            if layers and not self.silent:
                plural = "s" if len(layers) > 1 else ""
                print(f"Rendering {len(layers)} {group_name} layer{plural}...")
            for layer in sorted(layers, key=lambda group_layer: int(group_layer.zorder)):
                layer_artists = layer.render(self._ax, target_crs=self.target_crs)
                if layer_artists:
                    self._artists.track(layer_artists)

    def _apply_limits(self, external: set[Unit]) -> None:
        """Apply stored x/y limits to the axes, respecting external changes."""
        if self._xlim is None or self._ylim is None:
            self._ax.autoscale_view(
                scalex=self._xlim is None,
                scaley=self._ylim is None,
            )

        def apply_stored_xlim() -> None:
            if self._xlim is not None:
                self._ax.set_xlim(*self._xlim)

        def apply_stored_ylim() -> None:
            if self._ylim is not None:
                self._ax.set_ylim(*self._ylim)

        self._axes_state.reconcile(
            "x_limits",
            external,
            apply_stored_xlim,
            lambda: tuple(float(value) for value in self._ax.get_xlim()),
        )
        self._axes_state.reconcile(
            "y_limits",
            external,
            apply_stored_ylim,
            lambda: tuple(float(value) for value in self._ax.get_ylim()),
        )

    def _apply_axis_visibility(self, external: set[Unit]) -> None:
        def apply_visibility() -> None:
            if self._show_axis:
                self._ax.set_axis_on()
            else:
                self._ax.set_axis_off()

        self._axes_state.reconcile(
            "axis_visibility", external, apply_visibility, lambda: bool(self._ax.axison)
        )

    def _apply_aspect(
        self,
        external: set[Unit],
        previous: float | Literal["auto", "equal"],
    ) -> None:
        """Preserve an externally selected aspect across GeoPandas layer rendering."""
        if "aspect" in external:
            self._ax.set_aspect(previous)
        else:
            self._axes_state.record_default("aspect", self._ax.get_aspect())

    def _apply_extra_units(self, external: set[Unit]) -> None:
        """Hook: reconcile subclass-managed axes units (e.g. the dot-density legend)."""

    def _build_and_apply_settings(self) -> dict[str, Point]:
        """Snapshot → remove gerrytools artists → rebuild → apply settings.

        Deferred labels draw last, after limits are applied, so their representative
        points and clipping reflect the final view.
        """
        before, external = self._axes_state.begin_rebuild(self._ax)
        previous_aspect = cast(float | Literal["auto", "equal"], self._ax.get_aspect())
        self._artists.remove_all()
        self._recompute_data_limits()
        self._build_plot()
        self._axes_state.restore_autoscale_protected(self._ax, before, external)
        self._apply_text(self._title_text, external)
        self._apply_axis_visibility(external)
        self._apply_aspect(external, previous_aspect)
        self._apply_limits(external)
        self._apply_extra_units(external)
        label_positions = _draw_deferred_labels(
            self._label_requests,
            ax=self._ax,
            target_crs=self.target_crs,
            artists=self._artists,
        )
        self._last_label_positions = label_positions
        return label_positions

    def get_label_positions(self, *, as_lat_long: bool = False) -> tuple[str, dict[str, Point]]:
        """Get computed label positions from the current plot build.

        Reuses the positions computed by the most recent build when one has happened;
        otherwise triggers a build first.

        Args:
            as_lat_long (bool, optional): Whether to convert points to ``EPSG:4326``. Defaults to
                False.

        Returns:
            tuple[str, dict[str, Point]]: CRS string and label-to-point mapping. If separate
                layers render the same label text, the last layer's position is returned.

        Raises:
            ValueError: If ``as_lat_long`` is True on a plot with no CRS.
            RuntimeError: If a plot build completes without producing label positions.
        """
        self._update_axis()
        positions = self._last_label_positions
        if positions is None:  # pragma: no cover - successful builds always assign this
            raise RuntimeError("GeoPlot build completed without label positions.")
        label_points = GeoSeries(positions, crs=self.target_crs)
        if as_lat_long:
            if label_points.crs is None:
                raise ValueError(
                    "get_label_positions(as_lat_long=True) cannot reproject positions on a "
                    "plot with no CRS: use a base gdf with a CRS (or set target_crs).",
                )
            label_points = label_points.to_crs("EPSG:4326")
        return (
            str(label_points.crs.to_string() if label_points.crs is not None else "undefined"),
            {str(label): Point(pt.x, pt.y) for label, pt in label_points.items()},
        )
