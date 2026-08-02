import dataclasses
import hashlib
import math
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from warnings import warn

import numpy as np
from geopandas import GeoDataFrame
from matplotlib.colors import to_hex, to_rgba
from matplotlib.lines import Line2D
from numpy.random import Generator
from pandas.api.types import is_numeric_dtype

from gerrytools.colors import resolve_color_and_alpha
from gerrytools.logging import get_logger
from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting._axes_state import Unit
from gerrytools.plotting._legend_mixin import _LegendMixin
from gerrytools.plotting._rng import resolve_numpy_rng
from gerrytools.plotting.data.options import DEFAULT_EDGE_WIDTH, _needs_default_edge_width
from gerrytools.plotting.geometry._dot_sampling import _make_random_points
from gerrytools.plotting.geometry._labels import LabelOptions
from gerrytools.plotting.geometry._layers._base import _to_target_crs
from gerrytools.plotting.geometry.geoplotbase import GeoPlotBase
from gerrytools.plotting.mpl.label_text_options import LabelBoxOptions, LabelFontOptions
from gerrytools.plotting.mpl.legend_options import LegendOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.plotting.utils import _replace_with_color_overrides
from gerrytools.typing import UNSET, Color, CRSLike, LegendHandle, Unset

logger = get_logger(__name__)

_SCATTER_BLOCK = 200_000

# Default label styling for the district outlines: bold numbers in wheat circles.
_DEFAULT_DD_LABEL_FONT = LabelFontOptions(
    fontfamily="sans-serif",
    fontsize=4,
    fontweight="bold",
    fontcolor="black",
    outlinecolor="none",
)
_DEFAULT_DD_LABEL_BOX = LabelBoxOptions(
    enabled=True,
    boxstyle="circle",
    pad=0.5,
    facecolor="#f1deb8",
    facealpha=1.0,
    edgecolor="black",
    edgealpha=1.0,
    edgewidth=0.5,
)


@dataclass(frozen=True, slots=True)
class _DensityLayerRecord:
    """One dot-density layer: data column, dot color, cache location, and sampling identity.

    ``people_per_dot`` (also embedded in the cache path) is the value in effect when the
    layer was added, so it is captured exactly once here rather than re-derived at draw
    time. ``insertion_index`` pins the layer's sampling stream: each layer samples from a
    generator derived from the plot's seed root plus (column, people_per_dot,
    insertion_index), so draws for other layers or rebuilds cannot shift its dots.
    ``n_jobs`` / ``n_chunks`` are kept so an invalidated cache regenerates with the same
    parallelization the caller asked for (they never affect dot positions).
    """

    column: str
    color: str
    cache_path: Path
    people_per_dot: int | float
    insertion_index: int
    n_jobs: int
    n_chunks: int | np.integer
    source_fingerprint: bytes


class DotDensityPlot(_LegendMixin, GeoPlotBase):
    """Class for creating dot density plots from GeoDataFrames.

    Each dot represents a specified number of people, and dots are randomly placed
    within the polygons of the GeoDataFrame. Different data columns can be visualized
    with different colors.

    Attributes:
        gdf (GeoDataFrame): The base GeoDataFrame for the plot.
        fig (Figure): The Matplotlib Figure object.
        target_crs (CRSLike | None): The target CRS for reprojecting geometries.
        silent (bool): Whether to suppress informational output throughout
            the rendering process.
        show_legend (bool): Whether to show the legend.
    """

    def __init__(
        self,
        gdf: GeoDataFrame,
        *,
        outline_column: str,
        dpi: int | None = None,
        title: str | None = None,
        show_axis: bool = False,
        target_crs: CRSLike | None = None,
        default_outline: bool = False,
        silent: bool = False,
        people_per_dot: int = 100,
        show_labels: bool = True,
        label_options: LabelOptions | None = None,
        show_legend: bool = False,
        edgecolor: Color | None = "black",
        edgealpha: float | None = None,
        edgewidth: float = 0.6,
        rng_seed: int | None = None,
        rng: Generator | None = None,
    ) -> None:
        """Initialize a DotDensityPlot instance.

        By default, dot density plots will include an outline layer based on the
        `outline_column` provided. This is used to show the boundaries of districts or
        other relevant areas to help provide context for the dot density visualization.

        Args:
            gdf (GeoDataFrame): The base GeoDataFrame for the plot.
            outline_column (str): The column in the GeoDataFrame to use for outlining
                districts or areas.
            people_per_dot (int, optional): Number of people represented by each dot.
                Defaults to 100.
            show_labels (bool, optional): Whether to show labels for the outlined areas.
                Defaults to True.
            label_options (LabelOptions | None, optional): Bundled label styling and
                placement options (label style or font/box options, per-label adjustments and
                font sizes, and excluded labels). Unset ``font_options`` / ``box_options`` (with
                no label style) fall back to the dot-density defaults: bold sans-serif numbers in
                wheat circles. Defaults to None.
            edgecolor (Color | None, optional): Color of the outline edges. Pass ``None``
                for no outline. Defaults to 'black'.
            edgealpha (float | None, optional): Alpha transparency for the outline edges.
                Defaults to None.
            edgewidth (float, optional): Width of the outline edges. Defaults to 0.6.
            dpi (int, optional): Dots per inch for the plot. Defaults to 300.
            title (str | None, optional): The title of the plot. Defaults to None.
            show_axis (bool, optional): Whether to show the axis. Defaults to False.
            target_crs (CRSLike | None, optional): Target CRS for reprojecting geometries.
                Defaults to None.
            default_outline (bool, optional): Whether to include a default outline
                layer. Defaults to False because the outline layer is already being added.
            silent (bool, optional): Whether to suppress informational output throughout
                the rendering process. Defaults to False.
            show_legend (bool, optional): Whether to show the legend. Defaults to False.
            rng_seed (int | None, optional): Seed for reproducible dot placement. Each density
                layer and the rebuild-time interleaving draw from generators derived from this
                seed, so layers are independent of each other and rebuilds are stable.
                Defaults to None.
            rng (Generator | None, optional): Explicit NumPy generator used (once) to derive
                the seed root instead of constructing one from ``rng_seed``. Defaults to None.
        """
        if target_crs is not None and gdf.crs is not None:
            gdf = gdf.to_crs(target_crs)

        super().__init__(
            gdf=gdf,
            dpi=dpi,
            title=title,
            show_axis=show_axis,
            target_crs=target_crs,
            default_outline=default_outline,
            silent=silent,
        )
        resolved_rng, self._rng_seed = resolve_numpy_rng(
            seed=rng_seed, rng=rng, field_name="rng_seed"
        )
        self._seed_root: int = self._derive_seed_root(resolved_rng)
        self.people_per_dot = people_per_dot

        # Unset font/box pieces (without a label style) fall back to the dot-density defaults.
        if label_options is None:
            label_options = LabelOptions(
                font_options=_DEFAULT_DD_LABEL_FONT,
                box_options=_DEFAULT_DD_LABEL_BOX,
            )
        elif label_options.label_style is None:
            label_options = dataclasses.replace(
                label_options,
                font_options=label_options.font_options or _DEFAULT_DD_LABEL_FONT,
                box_options=label_options.box_options or _DEFAULT_DD_LABEL_BOX,
            )

        # outlines for districts
        self.add_outline_layer(
            dissolve_column=outline_column,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            show_labels=show_labels,
            label_options=label_options,
            zorder=100,
        )

        self._marker_options = PointMarkerOptions(
            marker="o",
            markersize=1.0,
            markerfacecolor="none",
            markerfacealpha=None,
            markeredgecolor="none",
            markeredgealpha=None,
            markeredgewidth=0.0,
        )

        # Used for caching the dots so that you can iterate quickly when adjusting styles
        # TemporaryDirectory's own finalizer removes the directory when the plot is collected.
        self._temp_dir = tempfile.TemporaryDirectory()

        logger.debug(f"Created temporary directory for dot density plot: {self._temp_dir.name}")
        self._temp_dir_name = str(self._temp_dir.name)
        self._density_layers: dict[str, _DensityLayerRecord] = {}

        self._legend_options = LegendOptions()
        # Constructor default (False) is "no opinion"; only a truthy request claims the
        # legend unit, mirroring the data family's constructor-arg reclaim semantics.
        self._show_legend: bool = False
        if show_legend:
            self.show_legend = True

    @property
    def show_legend(self) -> bool:
        """Whether to render the dot-density legend on rebuild."""
        return self._show_legend

    @show_legend.setter
    @deferred_axis_update
    def show_legend(self, value: bool) -> None:
        self._show_legend = bool(value)
        # Legend identity is recorded by ``_apply_legend`` at the next rebuild.
        self._axes_state.reclaim_without_value("legend")

    def _derive_seed_root(self, resolved_rng: Generator) -> int:
        """Fixed root for every derived generator; drawn once when no seed was given.

        A fixed root keeps seedless plots stable too: dots do not move across rebuilds.
        """
        if self._rng_seed is not None:
            return self._rng_seed
        return int(resolved_rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))

    def _layer_rng(self, record: _DensityLayerRecord) -> Generator:
        """Sampling generator for one density layer.

        Derived from the seed root plus the layer's identity, so no other draw (other
        layers, rebuilds) can shift its dots.
        """
        layer_key = f"{record.column}|{record.people_per_dot!r}|{record.insertion_index}"
        return np.random.default_rng([zlib.crc32(layer_key.encode("utf-8")), self._seed_root])

    def _invalidate_dot_caches(self) -> None:
        """Delete cached dot files so the next build resamples every density layer."""
        for record in self._density_layers.values():
            record.cache_path.unlink(missing_ok=True)

    @property
    def rng_seed(self) -> int | None:
        """The configured seed, or None for a random root stable across rebuilds."""
        return self._rng_seed

    @rng_seed.setter
    @deferred_axis_update
    def rng_seed(self, seed: int | None) -> None:
        """Set the seed; assigning the current value is idempotent."""
        if seed is None and self._rng_seed is None:
            return
        resolved_rng, self._rng_seed = resolve_numpy_rng(seed=seed, field_name="rng_seed")
        self._seed_root = self._derive_seed_root(resolved_rng)
        self._invalidate_dot_caches()

    @property
    def target_crs(self) -> CRSLike | None:
        """Coordinate reference system used to render geometry layers."""
        return self._target_crs

    @target_crs.setter
    @deferred_axis_update
    def target_crs(self, value: CRSLike | None) -> None:
        """Set the render CRS, invalidating cached dots so they are resampled in it.

        Outline layers reproject at render time, but dots are cached in the CRS they were
        sampled in; without invalidation a CRS change would draw the stale coordinates.
        """
        self._set_target_crs(value)
        self._invalidate_dot_caches()

    @property
    def people_per_dot(self) -> int | float:
        """Number of people represented by each dot.

        Applies to density layers added after the change; already-added layers keep the
        value captured when they were added.

        Raises:
            ValueError: When assigned a value that is not positive and finite.
        """
        return self._people_per_dot

    @people_per_dot.setter
    def people_per_dot(self, value: int | float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"people_per_dot must be a positive finite number, but found {value!r}."
            )
        self._people_per_dot = value

    @deferred_axis_update
    def set_marker_options(
        self,
        marker_options: PointMarkerOptions | None = None,
        *,
        marker: str | None = None,
        markersize: float | None = None,
        markeredgecolor: Color | None | Unset = UNSET,
        markeredgealpha: float | None = None,
        markeredgewidth: float | None = None,
    ) -> None:
        """Set global marker options for all dot density layers.

        This method sets the marker style for all dot density layers in the plot,
        so all dots share the same marker style except for color (which is set
        per-layer when adding a dot density layer).

        Mirrors the styling-pattern used everywhere else in the package: pass a
        pre-built :class:`PointMarkerOptions` for compose-and-reuse, or provide
        individual kwargs (which override the corresponding fields on the
        passed-in options). All-defaults call gives a sensible 1pt dot.

        Args:
            marker_options (PointMarkerOptions | None, optional): Pre-built styling.
                Any styling kwarg passed explicitly overrides the corresponding
                field on ``marker_options``. Defaults to None.
            marker (str, optional): Marker style. Defaults to None, which uses
                ``marker_options`` or ``"o"``.
            markersize (float, optional): Marker size. Defaults to None, which uses
                ``marker_options`` or 1.0.
            markeredgecolor (Color | None, optional): The color of the marker edges.
                Pass ``None`` for no edge. If omitted, uses ``marker_options`` when provided or
                the edgeless default otherwise.
            markeredgealpha (float | None, optional): Edge opacity. Defaults to None, which uses
                the alpha encoded in ``markeredgecolor``.
            markeredgewidth (float, optional): The width of the marker edges. Selecting a visible
                edge color while omitting the width uses the package's default visible width.
                Pass ``0`` explicitly to keep the edge hidden.
        """
        # The dot-density global default: tiny edgeless dot.
        base = (
            marker_options
            if marker_options is not None
            else PointMarkerOptions(
                marker="o",
                markersize=1.0,
                markeredgecolor="none",
                markeredgewidth=0.0,
            )
        )
        resolved_options = _replace_with_color_overrides(
            base,
            ("markeredgecolor", "markeredgealpha"),
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgealpha=markeredgealpha,
            markeredgewidth=markeredgewidth,
        )
        if _needs_default_edge_width(
            edgewidth_given=markeredgewidth is not None,
            resolved_edgewidth=resolved_options.markeredgewidth,
            resolved_edgecolor=(
                resolved_options.markeredgecolor if not isinstance(markeredgecolor, Unset) else None
            ),
        ):
            resolved_options = dataclasses.replace(
                resolved_options,
                markeredgewidth=DEFAULT_EDGE_WIDTH,
            )
        self._marker_options = resolved_options

    @deferred_axis_update
    def add_density_layer(
        self,
        column: str,
        color: Color | None,
        *,
        refresh_cache: bool = False,
        n_jobs: int = -1,
        n_chunks: int | np.integer = 10,
    ) -> None:
        """Add a dot density layer for a specific data column.

        This method generates random dots within the polygons of the GeoDataFrame
        based on the values in the specified data column. Each dot represents a
        certain number of people, defined by `people_per_dot`. The dots are colored
        according to the specified color.

        The Point objects generated are cached in a temporary directory to speed up
        subsequent renderings. If the same column and color are requested again, the
        cached dots will be used unless `refresh_cache` is set to True.

        Args:
            column (str): The name of the data column to visualize.
            color (Color | None): The color of the dots. Pass ``None`` for transparent
                dots.
            refresh_cache (bool, optional): If True, forces regeneration of cached dots.
                Defaults to False.
            n_jobs (int, optional): Number of parallel jobs to use for processing when
                generating dots. Defaults to -1 which will use all available cores minus two.
            n_chunks (int, optional): Number of chunks used to split polygon processing work.
                Defaults to ``10``.

        Raises:
            ValueError: If the column is missing, non-numeric, nonfinite, or contains negatives.
        """
        if column not in self.gdf.columns:
            raise ValueError(f"Column '{column}' not found in GeoDataFrame.")

        values = self.gdf[column]
        if not is_numeric_dtype(values):
            raise ValueError(f"Column '{column}' must be numeric to compute dot counts.")

        if any(values.isna()):
            raise ValueError(f"Column '{column}' contains NaN values.")

        if not np.isfinite(values).all():
            raise ValueError(f"Column '{column}' contains infinite values.")

        if any(values < 0):
            raise ValueError(f"Column '{column}' contains negative values.")

        resolved_color, resolved_alpha = resolve_color_and_alpha(color)
        color = (
            "none"
            if resolved_color == "none"
            else to_hex(
                to_rgba(resolved_color, resolved_alpha),
                keep_alpha=not math.isclose(resolved_alpha, 1.0),
            )
        )
        existing = self._density_layers.get(column)
        if existing is not None and existing.color == color and not refresh_cache:
            warn(
                f"Dots for column '{column}' with the same color already exist. "
                "Use 'refresh_cache=True' to recreate them.",
                UserWarning,
                stacklevel=1,
            )
            return

        insertion_index = (
            existing.insertion_index if existing is not None else len(self._density_layers)
        )
        # Dots are positional and color-independent, so a new color just recolors the cached
        # dots; the cache path is fixed when the layer is added.
        if existing is not None and not refresh_cache:
            cache_filepath = existing.cache_path
            layer_people_per_dot = existing.people_per_dot
        else:
            cache_filepath = (
                Path(self._temp_dir_name) / f"dots_{insertion_index}_ppd{self.people_per_dot}.npz"
            )
            layer_people_per_dot = self.people_per_dot
        record = _DensityLayerRecord(
            column=column,
            color=color,
            cache_path=cache_filepath,
            people_per_dot=layer_people_per_dot,
            # Re-adding a column keeps its slot, so its sampling stream is unchanged.
            insertion_index=insertion_index,
            n_jobs=n_jobs,
            n_chunks=n_chunks,
            source_fingerprint=self._density_fingerprint(column),
        )
        self._density_layers[column] = record

        if not cache_filepath.exists() or refresh_cache:
            self._generate_layer_dots(record)

    def _density_fingerprint(self, column: str) -> bytes:
        """Fingerprint the values and projected geometry that determine sampled dots."""
        sample_gdf = _to_target_crs(self.gdf, self.target_crs)
        digest = hashlib.blake2b(digest_size=16)
        for value, geometry in zip(sample_gdf[column], sample_gdf.geometry.to_wkb(), strict=True):
            for payload in (repr(value).encode(), geometry or b""):
                digest.update(len(payload).to_bytes(8, "little"))
                digest.update(payload)
        digest.update(str(sample_gdf.crs).encode())
        return digest.digest()

    def _generate_layer_dots(self, record: _DensityLayerRecord) -> None:
        """Sample and cache dots for one density layer, in the current target CRS."""
        if not self.silent:
            print(f"Generating dots for column '{record.column}'.")

        sample_gdf = _to_target_crs(self.gdf, self.target_crs)
        xs, ys, polyids = _make_random_points(
            gdf=sample_gdf.loc[:, [record.column, sample_gdf.geometry.name]],
            people_per_dot=record.people_per_dot,
            datacolumn_name=record.column,
            rng=self._layer_rng(record),
            n_jobs=record.n_jobs,
            n_chunks=record.n_chunks,
        )

        np.savez(record.cache_path, x=xs, y=ys, polyids=polyids)

    def _draw_interleaved_scatter_blocks(
        self,
        *,
        layers_xy_polyid: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        layer_colors: list[str],
    ) -> None:
        """Draw dots from all layers in interleaved blocks for visual mixing.

        Consumes ``layers_xy_polyid``: the list is cleared once the arrays are concatenated
        so the per-layer copies can be freed.

        Args:
            layers_xy_polyid (list[tuple[np.ndarray, np.ndarray, np.ndarray]]): Per-layer dot
                data as ``(x, y, polygon_id)`` arrays.
            layer_colors (list[str]): Per-layer marker colors.

        Returns:
            None
        """
        # Build one big table of points with polygon id and a layer id
        xs_all = np.concatenate([x for (x, _, _) in layers_xy_polyid])
        ys_all = np.concatenate([y for (_, y, _) in layers_xy_polyid])
        polyid_all = np.concatenate([polyid for (_, _, polyid) in layers_xy_polyid])

        # layer id per point -> used to map to colors fast
        layer_ids = np.concatenate(
            [np.full(len(x), i, dtype=np.int32) for i, (x, _, _) in enumerate(layers_xy_polyid)]
        )
        # The concatenated arrays are the working set now; free the per-layer copies.
        layers_xy_polyid.clear()

        palette = np.asarray(layer_colors, dtype=object)
        # Reseeded from the seed root on every call, so a rebuild never reshuffles the
        # visual interleaving (and never perturbs any other random stream).
        interleave_rng = np.random.default_rng([zlib.crc32(b"interleave"), self._seed_root])
        random_priority = interleave_rng.random(size=len(xs_all))

        # Randomize within each polygon: sort by (polyid, rnd)
        # Lexsort uses last key as primary sort key and avoids copying data
        order = np.lexsort((random_priority, polyid_all))
        del random_priority, polyid_all

        # Apply the permutation one array at a time so at most one extra full-length copy
        # is live; each reassignment frees the unsorted original.
        xs_all = xs_all[order]
        ys_all = ys_all[order]
        layer_ids = layer_ids[order]
        del order

        # Draw in blocks (keeps memory + scatter call size reasonable)
        n = len(xs_all)
        marker_settings = self._marker_options.to_mpl_scatter_settings_dict()

        for start in range(0, n, _SCATTER_BLOCK):
            end = min(n, start + _SCATTER_BLOCK)
            scatter_collection = self._ax.scatter(
                xs_all[start:end],
                ys_all[start:end],
                c=palette[layer_ids[start:end]],
                marker=marker_settings["marker"],
                s=marker_settings["s"],
                edgecolor=marker_settings["edgecolor"],
                linewidths=marker_settings["linewidths"],
                zorder=marker_settings["zorder"],
            )
            self._artists.track(scatter_collection)

    def _draw_all_dots(self) -> None:
        """Draw all dot density layers on the plot, regenerating any invalidated caches."""
        if len(self._density_layers) == 0:
            return
        layers_xy_polyid = []
        colors = []

        for column, record in list(self._density_layers.items()):
            fingerprint = self._density_fingerprint(record.column)
            if fingerprint != record.source_fingerprint:
                record = dataclasses.replace(record, source_fingerprint=fingerprint)
                self._density_layers[column] = record
                record.cache_path.unlink(missing_ok=True)
            if not record.cache_path.exists():
                self._generate_layer_dots(record)
            with np.load(record.cache_path) as np_data:
                layers_xy_polyid.append((np_data["x"], np_data["y"], np_data["polyids"]))
            colors.append(record.color)

        if not self.silent:
            columns = list(self._density_layers)
            plural = "s" if len(columns) > 1 else ""
            print(
                f"Rendering {sum(len(x) for x, _, _ in layers_xy_polyid):,} dots for "
                f"column{plural} '{', '.join(columns)}'..."
            )
        self._draw_interleaved_scatter_blocks(
            layers_xy_polyid=layers_xy_polyid, layer_colors=colors
        )

    def _dot_legend_handles(
        self,
        *,
        display_names: dict[str, str] | None = None,
        min_markersize: float | None = 6.0,
    ) -> list[LegendHandle]:
        """Build one legend handle per density layer.

        Args:
            display_names (dict[str, str] | None, optional): Mapping from column names to
                display names. Defaults to None.
            min_markersize (float | None, optional): Lower bound for the legend glyph size in
                points. Map dots are often well under a point wide, and a legend glyph that
                small is invisible. Defaults to None.

        Returns:
            list[LegendHandle]: One handle per density layer.
        """
        marker_settings = self._marker_options.to_mpl_scatter_settings_dict()
        legend_marker_size = float(np.sqrt(marker_settings["s"]))
        if min_markersize is not None:
            legend_marker_size = max(legend_marker_size, min_markersize)

        return [
            Line2D(
                [0],
                [0],
                label=display_names.get(record.column, record.column)
                if display_names is not None
                else record.column,
                linestyle="",
                markerfacecolor=record.color,
                marker=marker_settings["marker"],
                markersize=legend_marker_size,
                markeredgecolor=marker_settings["edgecolor"],
                markeredgewidth=marker_settings["linewidths"],
            )
            for record in self._density_layers.values()
        ]

    @property
    def _legend_handles(self) -> list[LegendHandle]:
        """One handle per density layer, with a readable minimum glyph size."""
        return self._dot_legend_handles()

    @property
    def _legend_enabled(self) -> bool:
        """Whether ``_apply_legend`` should place a legend; the mixin's enabled hook."""
        return self._show_legend

    def _apply_extra_units(self, external: set[Unit]) -> None:
        super()._apply_extra_units(external)
        self._apply_legend(external)

    def _build_plot(self) -> None:
        """Build the plot by rendering all layers and applying settings."""
        super()._build_plot()
        self._draw_all_dots()

    def save_legend(
        self,
        filepath: str,
        *,
        display_names: dict[str, str] | None = None,
        outer_padding: float = 0.07,
        dpi: int | None = None,
        **legend_kwargs: object,
    ) -> None:
        """Save the legend to a separate file.

        Args:
            filepath (str): The file path to save the legend to.
            display_names (dict[str, str] | None, optional): A mapping from original
                column names to new display names for the legend. If None, original column names
                are used. Defaults to None.
            dpi (int | None, optional): The DPI to use when saving the legend. If None, uses the
                same DPI as the main figure. Defaults to None.
            outer_padding (float, optional): The outer padding around the legend.
                Defaults to 0.07.
            **legend_kwargs (object): Additional keyword arguments passed to
                ``matplotlib.axes.Axes.legend``.

        Returns:
            None
        """
        if not self._density_layers:
            logger.warning("No density layers have been added, so there is no legend to save.")
            return

        self._save_legend_handles(
            self._dot_legend_handles(display_names=display_names),
            filepath,
            outer_padding=outer_padding,
            dpi=dpi,
            **legend_kwargs,
        )
