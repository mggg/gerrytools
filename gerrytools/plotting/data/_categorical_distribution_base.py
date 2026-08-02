from __future__ import annotations

import math
from abc import abstractmethod
from numbers import Real
from typing import Iterable, Iterator, Mapping, Sequence, TypeVar, cast, final

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from numpy.typing import NDArray

from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting.data._gerryplot_dataclasses import _PointSetData
from gerrytools.plotting.data.gerryplot import GerryPlotBase
from gerrytools.plotting.data.options import LineOptions, _FaceEdgeStyle
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions, _marker_legend_handle
from gerrytools.typing import Color, LegendHandle

MappedT = TypeVar("MappedT")


def _stringify_unique_labels(labels: Iterable[object]) -> list[str]:
    string_labels = [str(label) for label in labels]
    if len(set(string_labels)) != len(string_labels):
        raise ValueError("Category labels must be unique after conversion to strings.")
    return string_labels


class CategoricalDistributionPlotBase(GerryPlotBase):
    """Shared base for categorical distribution plots.

    This class centralizes label alignment, point-set overlays, category-center guides,
    and default x-axis tick behavior used by plots such as ``BoxPlot`` and ``ViolinPlot``.
    """

    # Human-readable dataset noun used in "No <noun>s added yet." build errors.
    _dataset_noun: str = "data set"

    def __init__(
        self,
        *,
        figure_size: tuple[float, float] | None,
        dpi: int | None,
        ax: Axes | None,
        legend: bool | None,
        xlabel: str | None,
        ylabel: str | None,
        title: str | None,
        group_width: float,
        width_scale: float,
    ) -> None:
        """Initialize shared categorical distribution plot state.

        Args:
            figure_size (tuple[float, float]): Figure size in inches.
            dpi (int): Figure resolution in dots per inch.
            ax (Axes | None): Existing axes to bind, or None to create a figure.
            legend (bool): Whether legend rendering is enabled.
            xlabel (str | None): X-axis label text. Defaults to None.
            ylabel (str | None): Y-axis label text. Defaults to None.
            title (str | None): Plot title text. Defaults to None.
            group_width (float): Width allocated to each category group on the x-axis.
                Must be in ``(0, 1]``.
            width_scale (float): Relative width scaling for grouped artists.
                Must be in ``(0, 1]``.

        Category-center guide lines start disabled; toggle them with
        :meth:`display_group_separators`.
        """
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            ax=ax,
            legend=legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        self.group_width = group_width
        self.width_scale = width_scale

        self._pointset_data_list: list[_PointSetData] = []
        self._labels: list[str] | None = None

        self._include_group_vlines = False
        self._group_vline_settings = LineOptions(
            linecolor="#cccccc",
            linealpha=1.0,
            linestyle="-",
            linewidth=0.8,
            zorder=-3,
        )

    @property
    def group_width(self) -> float:
        """Width allocated to each categorical group.

        Raises:
            ValueError: When assigned a value outside ``(0, 1]``.
        """
        return self._group_width

    @group_width.setter
    @deferred_axis_update
    def group_width(self, value: float) -> None:
        if value <= 0:
            raise ValueError("group_width must be positive.")
        if value > 1.0:
            raise ValueError("group_width must be <= 1.0 when centers are integers.")
        self._group_width = float(value)

    @property
    def width_scale(self) -> float:
        """Relative width of artists within each categorical slot.

        Raises:
            ValueError: When assigned a value outside ``(0, 1]``.
        """
        return self._width_scale

    @width_scale.setter
    @deferred_axis_update
    def width_scale(self, value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError("width_scale must be in (0.0, 1.0].")
        self._width_scale = float(value)

    @staticmethod
    def _convert_distribution_data_to_dictionary(
        scores: (
            Mapping[str, Sequence[float]]
            | Sequence[float]
            | Sequence[Sequence[float]]
            | pd.DataFrame
            | NDArray
        ),
        category_labels: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Convert distribution input to a dictionary mapping labels to score lists.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                Distribution values. DataFrames use each column as a category and dicts use
                their keys; list/array input is labeled by ``category_labels`` or auto-numbered.
            category_labels (list[str] | None, optional): Labels for list/array input. When None,
                categories are auto-numbered ``"0", "1", ...`` (0-indexed). Defaults to None.

        Returns:
            dict[str, list[float]]: Category label to score-list mapping.
        """

        def _to_labeled_dict(
            rows: list[list[float]],
            labels: list[str] | None,
        ) -> dict[str, list[float]]:
            """Pair score rows with labels, defaulting to ``"0".."k-1"`` when none given.

            Args:
                rows (list[list[float]]): One score list per category.
                labels (list[str] | None): Explicit category labels, or None to
                    auto-number the categories ``"0", "1", ...`` (0-indexed).

            Returns:
                dict[str, list[float]]: Category label to score-list mapping.

            Raises:
                ValueError: If ``labels`` is given but its length does not match the
                    number of score lists.
            """
            if labels is None:
                labels = [str(index) for index in range(len(rows))]
            elif len(labels) != len(rows):
                raise ValueError(
                    f"category_labels has length {len(labels)} but you provided "
                    f"{len(rows)} score lists."
                )
            string_labels = _stringify_unique_labels(labels)
            return dict(zip(string_labels, rows, strict=True))

        if isinstance(scores, Mapping):
            typed_scores = cast("Mapping[str, Sequence[float]]", scores)
            return _to_labeled_dict(
                [list(values) for values in typed_scores.values()], list(typed_scores)
            )

        if isinstance(scores, pd.DataFrame):
            if not scores.columns.is_unique:
                raise ValueError("Category labels must be unique after conversion to strings.")
            return _to_labeled_dict(
                [scores.iloc[:, index].dropna().tolist() for index in range(scores.shape[1])],
                list(scores.columns),
            )

        if isinstance(scores, Sequence):
            if len(scores) == 0:
                raise ValueError("scores is empty; provide at least one score list.")

            first = scores[0]

            def _is_scalar(x: object) -> bool:
                """Return whether an object should be treated as a scalar score value.

                Args:
                    x (object): Candidate object.

                Returns:
                    bool: True when ``x`` is scalar-like for score parsing.
                """
                return isinstance(x, (str, bytes, Real, np.generic))

            def _is_score_series(x: object) -> bool:
                """Return whether an object should be treated as a score sequence.

                Args:
                    x (object): Candidate object.

                Returns:
                    bool: True when ``x`` behaves like a sequence of scores.
                """
                if _is_scalar(x):
                    return False
                return isinstance(x, (list, tuple, np.ndarray, pd.Series))

            if _is_score_series(first):
                nested = cast(Sequence[Sequence[float]], scores)
                rows = [list(score_list) for score_list in nested]
            else:
                # A flat sequence of scalars is a single category.
                flat = cast(Sequence[float], scores)
                rows = [list(flat)]
            return _to_labeled_dict(rows, category_labels)

        if isinstance(scores, np.ndarray):
            if scores.ndim == 1:
                # A flat array of scalars is a single category, just like a flat list.
                rows = [[float(score) for score in scores]]
            elif scores.ndim == 2:
                rows = [[float(score) for score in row] for row in scores]
            else:
                raise ValueError("scores array must be 1D or 2D.")
            return _to_labeled_dict(rows, category_labels)

        raise TypeError(
            "Scores must be a dict[str, list[float]], list[float], list[list[float]], "
            "or pd.DataFrame."
        )

    def _sync_labels(
        self,
        incoming: list[str],
        *,
        add_extra_labels: bool,
        item_name: str,
    ) -> None:
        """Validate and update plot labels for an incoming dataset.

        Args:
            incoming (list[str]): Label sequence extracted from the incoming dataset.
            add_extra_labels (bool): Whether to extend existing labels with unseen values.
            item_name (str): Dataset descriptor used in validation messages.

        Returns:
            None

        Raises:
            ValueError: If labels differ and ``add_extra_labels`` is False.
        """
        if self._labels is None:
            self._labels = incoming
            return

        if incoming != self._labels:
            if not add_extra_labels:
                raise ValueError(
                    f"{item_name} labels must match existing labels in the same order.\n"
                    f"Expected: {self._labels}\nGot:      {incoming}\n"
                    "If you want to allow additional labels, set add_extra_labels=True."
                )
            self._labels = list(dict.fromkeys(self._labels + incoming))

    def _convert_pointset_to_dict(
        self,
        values: dict[str, float] | list[float] | pd.Series | pd.DataFrame,
        labels: list[str] | None = None,
        *,
        column: str | None = None,
    ) -> dict[str, float]:
        """Convert point-set input to a dictionary mapping labels to float values.

        Args:
            values (dict[str, float] | list[float] | pd.Series | pd.DataFrame): Point values.
            labels (list[str] | None, optional): Labels for list-based input. Defaults to None.
            column (str | None, optional): DataFrame column to read when ``values`` is a
                DataFrame with more than one column. Defaults to None.

        Returns:
            dict[str, float]: Category label to point value mapping.
        """
        if isinstance(values, dict):
            return dict(
                zip(
                    _stringify_unique_labels(values),
                    map(float, values.values()),
                    strict=True,
                )
            )

        if isinstance(values, pd.Series):
            return dict(
                zip(
                    _stringify_unique_labels(values.index),
                    map(float, values),
                    strict=True,
                )
            )

        if isinstance(values, pd.DataFrame):
            if column is None:
                if values.shape[1] != 1:
                    raise ValueError(
                        "DataFrame pointset input must have exactly one column, or pass column=..."
                    )
                ser = values.iloc[:, 0]
            else:
                if column not in values.columns:
                    raise ValueError(f"column {column!r} not found in DataFrame.")
                ser = values[column]
            return dict(
                zip(
                    _stringify_unique_labels(ser.index),
                    map(float, ser),
                    strict=True,
                )
            )

        vals = list(values)
        if labels is None:
            if self._labels is None:
                raise ValueError(
                    "For list pointset input, provide labels=... (or add distributions first to "
                    "define labels)."
                )
            labels = self._labels

        if len(vals) != len(labels):
            raise ValueError(
                f"Point set values length {len(vals)} does not match labels length {len(labels)}."
            )
        return dict(zip(_stringify_unique_labels(labels), map(float, vals), strict=True))

    def add_pointset(
        self,
        values: dict[str, float] | list[float] | pd.Series | pd.DataFrame,
        *,
        labels: list[str] | None = None,
        column: str | None = None,
        name: str | None = None,
        facecolor: Color | None = "black",
        facealpha: float | None = None,
        marker: str = "o",
        markersize: float = 7.0,
        markeredgecolor: Color | None = "black",
        markeredgealpha: float | None = None,
        markeredgewidth: float = 0.8,
        x_offset: float | None = None,
        zorder: int = 2,
        add_extra_labels: bool = False,
    ) -> None:
        """Add a marker point set across categorical x positions.

        Args:
            values (dict[str, float] | list[float] | pd.Series | pd.DataFrame): Point values.
            labels (list[str] | None, optional): Labels for list-based input. Defaults to None.
            column (str | None, optional): DataFrame column to read when ``values`` is a
                DataFrame with more than one column. Defaults to None.
            name (str | None, optional): Legend/display label for this point set.
                Defaults to None.
            facecolor (Color | None, optional): Marker face color. Pass ``None`` for no
                fill. Defaults to ``"black"``.
            facealpha (float | None, optional): Marker face alpha in ``[0, 1]``. Defaults to None.
            marker (str, optional): Marker style passed to Matplotlib. Defaults to ``"o"``.
            markersize (float, optional): Marker size. Defaults to ``7.0``.
            markeredgecolor (Color | None, optional): Marker edge color. Pass ``None``
                for no edge. Defaults to ``"black"``.
            markeredgealpha (float | None, optional): Marker edge alpha in ``[0, 1]``.
                Defaults to None.
            markeredgewidth (float, optional): Marker edge width. Defaults to ``0.8``.
            x_offset (float | None, optional): Explicit horizontal offset from the category
                center. If None, offsets are assigned automatically. Defaults to None.
            zorder (int, optional): Matplotlib z-order. Defaults to ``2``.
            add_extra_labels (bool, optional): If True, allows this point set to introduce
                additional labels beyond the current label set. Defaults to ``False``.

        Raises:
            ValueError: If ``x_offset`` is not finite.
        """
        if x_offset is not None:
            x_offset = float(x_offset)
            if not math.isfinite(x_offset):
                raise ValueError(f"x_offset must be finite; got {x_offset!r}.")
        values_dict = self._convert_pointset_to_dict(values, labels, column=column)
        incoming = list(values_dict.keys())
        self._sync_labels(incoming, add_extra_labels=add_extra_labels, item_name="point set")

        set_name = name or f"Point Set {len(self._pointset_data_list) + 1}"
        self._pointset_data_list.append(
            _PointSetData(
                name=set_name,
                values_dict=values_dict,
                point_data=PointMarkerOptions(
                    markerfacecolor=facecolor,
                    markerfacealpha=facealpha,
                    marker=marker,
                    markersize=markersize,
                    markeredgecolor=markeredgecolor,
                    markeredgealpha=markeredgealpha,
                    markeredgewidth=markeredgewidth,
                    zorder=zorder,
                ),
                x_offset=x_offset,
            )
        )
        self._claim_legend_if_named(name)

    @deferred_axis_update
    def update_group_vline_settings(
        self,
        *,
        linecolor: Color | None = "#cccccc",
        linealpha: float = 1.0,
        linestyle: str = "-",
        linewidth: float = 0.8,
        zorder: int = -3,
    ) -> None:
        """Update vertical guide-line style for category centers.

        Calling this also enables the separators, as if by
        ``display_group_separators(True)``; use that method to toggle them without
        restyling.

        Args:
            linecolor (Color | None, optional): Guide line color. Pass ``None`` for
                transparent lines. Defaults to ``"#cccccc"``.
            linealpha (float, optional): Guide line alpha in ``[0, 1]``. Defaults to ``1.0``.
            linestyle (str, optional): Matplotlib line style. Defaults to ``"-"``.
            linewidth (float, optional): Guide line width. Defaults to ``0.8``.
            zorder (int, optional): Matplotlib z-order. Defaults to ``-3``.
        """
        self._include_group_vlines = True
        self._group_vline_settings = LineOptions(
            linecolor=linecolor,
            linealpha=linealpha,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )

    @deferred_axis_update
    def clear_verticals(self) -> None:
        """Clear all vertical overlays and disable category-center guide lines."""
        self._include_group_vlines = False
        self._annotations.clear_verticals()

    @deferred_axis_update
    def display_group_separators(self, enabled: bool) -> None:
        """Set whether vertical category-group separators are displayed.

        Args:
            enabled (bool): Whether to display the separators.
        """
        self._include_group_vlines = enabled

    def _grouped_layout(self, n_sets: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute the side-by-side grouped layout for ``n_sets`` datasets.

        Returns:
            tuple[np.ndarray, np.ndarray, float]: Category centers, the per-set x offsets
            from those centers, and the width each drawn artist should use.
        """
        centers = self._category_centers
        slot_width = self.group_width / n_sets
        offsets = (np.arange(n_sets) - (n_sets - 1) / 2.0) * slot_width
        artist_width = slot_width * self.width_scale
        return centers, offsets, artist_width

    def _present_positions(
        self,
        mapping: Mapping[str, MappedT],
        positions: np.ndarray,
    ) -> Iterator[tuple[str, MappedT, float]]:
        """Yield ``(label, value, x)`` triples for the categories present in ``mapping``.

        Iterates the plot's labels in order, pairing each with its position and skipping
        categories the mapping does not provide.
        """
        assert self._labels is not None
        for label, x in zip(self._labels, positions, strict=True):
            if label in mapping:
                yield label, mapping[label], float(x)

    def _patch_legend_handles(
        self, entries: Iterable[tuple[str, _FaceEdgeStyle]]
    ) -> list[LegendHandle]:
        """Build one Patch legend handle per ``(name, style)`` dataset entry."""
        return [
            Patch(
                facecolor=self._resolved_rgba(
                    style.facecolor,
                    style.facealpha,
                    field="facecolor",
                ),
                edgecolor=self._resolved_rgba(
                    style.edgecolor,
                    style.edgealpha,
                    field="edgecolor",
                ),
                linewidth=style.edgewidth,
                label=name,
            )
            for name, style in entries
        ]

    @property
    @abstractmethod
    def _datasets(self) -> Sequence[object]:
        """The plot's stored datasets, used for the empty-plot guard."""

    @abstractmethod
    def _draw_datasets(self) -> None:
        """Draw this plot's datasets onto the axes."""

    @final
    def _build_plot(self) -> None:
        """Build the figure: guards, datasets, point sets, then optional group guides."""
        if self._labels is None or len(self._labels) == 0:
            raise ValueError("No labels defined yet.")
        if len(self._datasets) == 0:
            raise ValueError(f"No {self._dataset_noun}s added yet.")

        self._draw_datasets()
        self._draw_pointset(self._category_centers)
        if self._include_group_vlines:
            self._draw_group_vlines()

    def _default_x_tick_locations(self) -> list[float] | None:
        """Get default x-tick locations at the center of each category group."""
        return list(self._category_centers)

    def _default_x_tick_labels(self, tick_locations: list[float]) -> list[str] | None:
        """Place category labels only at their category centers.

        Args:
            tick_locations (list[float]): Candidate x-tick positions.

        Returns:
            list[str] | None: One label or an empty string for each tick.
        """
        if (
            self._labels is None
        ):  # pragma: no cover - _labels is always set before tick helpers are called
            return None  # pragma: no cover
        labels = []
        for location in tick_locations:
            category = int(location) - 1
            labels.append(
                self._labels[category]
                if location == category + 1 and 0 <= category < len(self._labels)
                else ""
            )
        return labels

    def _draw_group_vlines(self) -> None:
        """Draw vertical guide lines at each category center."""
        for x in self._category_centers:
            line = self._ax.axvline(
                x,
                color=self._resolved_rgba(
                    self._group_vline_settings.linecolor,
                    self._group_vline_settings.linealpha,
                    field="linecolor",
                ),
                linestyle=self._group_vline_settings.linestyle,
                linewidth=self._group_vline_settings.linewidth,
                zorder=self._group_vline_settings.zorder,
            )
            self._artists.track(line)

    @property
    def _category_centers(self) -> np.ndarray:
        """Calculate x-axis centers for each category."""
        if self._labels is None:
            return np.array([])
        n_categories = len(self._labels)
        return 1.0 + np.arange(n_categories, dtype=float)

    def _draw_pointset(self, centers: np.ndarray) -> None:
        """Draw all configured point sets on the plot.

        Args:
            centers (np.ndarray): X-axis category centers.

        Returns:
            None
        """
        if len(self._pointset_data_list) == 0 or self._labels is None:
            return

        n = len(self._pointset_data_list)
        span = min(self.group_width * 0.8, 0.8)

        auto_offsets = (np.arange(n) - (n - 1) / 2.0) * (span / n)

        for i, sdata in enumerate(self._pointset_data_list):
            offset = float(sdata.x_offset) if sdata.x_offset is not None else float(auto_offsets[i])
            xs = centers + offset

            ys = np.array([sdata.values_dict.get(lab, np.nan) for lab in self._labels], dtype=float)
            mask = np.isfinite(ys)
            if not np.any(mask):
                continue
            point_lines = self._ax.plot(
                xs[mask],
                ys[mask],
                linestyle="none",
                clip_on=True,
                rasterized=True,
                **sdata.point_data.to_mpl_settings_dict(),
            )
            self._artists.track(point_lines)

    def _pointset_legend_handles(self) -> list[LegendHandle]:
        """Generate legend handles for all point sets."""
        return [
            _marker_legend_handle(sdata.point_data, sdata.name)
            for sdata in self._pointset_data_list
        ]
