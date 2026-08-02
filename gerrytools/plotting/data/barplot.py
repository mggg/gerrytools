from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from gerrytools.logging import get_logger
from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting.data._categorical_distribution_base import CategoricalDistributionPlotBase
from gerrytools.plotting.data.options import BarPlotOptions
from gerrytools.plotting.utils import UNSET, Unset, _coerce_to_1d_finite_float_array
from gerrytools.typing import Color, LegendHandle, NumericArrayLike

logger = get_logger(__name__)


@dataclass(frozen=True)
class _BarSetData:
    """A set of bars to render for one model/series (one bar per category)."""

    name: str
    heights_dict: dict[str, float]
    style: BarPlotOptions

    def __post_init__(self) -> None:
        for label, height in self.heights_dict.items():
            if not math.isfinite(float(height)):
                raise ValueError(
                    f"_BarSetData {self.name!r}: height for category {label!r} must be finite; "
                    f"got {height!r}."
                )
        object.__setattr__(
            self,
            "heights_dict",
            {str(label): float(height) for label, height in self.heights_dict.items()},
        )


class BarPlot(CategoricalDistributionPlotBase):
    """Create categorical bar chart figures with spaced bars.

    Unlike :class:`Histogram`, which bins continuous values onto a shared numeric axis,
    a ``BarPlot`` places one bar per named category with visible gaps between categories
    (controlled by ``group_width``) and between bars within a group (controlled by
    ``width_scale``). Heights can be supplied directly via :meth:`add_dataset`
    or counted from raw values via :meth:`add_counts_dataset`.

    Multiple datasets share the category axis: they are drawn side by side within each
    category group by default, or on top of one another when ``stacked=True``.
    """

    _dataset_noun = "bar set"

    def __init__(
        self,
        *,
        figure_size: tuple[float, float] | None = None,
        dpi: int | None = None,
        ax: Axes | None = None,
        legend: bool | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        width_scale: float = 0.8,
        group_width: float = 0.7,
        stacked: bool = False,
    ) -> None:
        """Initialize a BarPlot.

        Args:
            figure_size (tuple[float, float] | None, optional): Figure size in inches. Defaults to
                Matplotlib's setting.
            dpi (int | None, optional): Figure resolution in dots per inch. Defaults to
                Matplotlib's setting.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing Axes
                instead of creating a fresh figure. Defaults to None.
            legend (bool | None, optional): Whether to include a legend. None leaves existing axes
                state unchanged. Defaults to None.
            xlabel (str | None, optional): X-axis label text. Defaults to None.
            ylabel (str | None, optional): Y-axis label text. Defaults to None.
            title (str | None, optional): Plot title text. Defaults to None.
            width_scale (float, optional): Relative width of each bar within its slot;
                values below 1 leave gaps between bars in a group. Defaults to ``0.8``.
            group_width (float, optional): Width allocated to each category group;
                values below 1 leave gaps between categories. Defaults to ``0.7``.
            stacked (bool, optional): If True, datasets are stacked on top of one another
                at each category instead of drawn side by side. Defaults to False.

        Toggle the per-group vertical guide lines with :meth:`display_group_separators` after
        construction.
        """
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            ax=ax,
            legend=legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            group_width=group_width,
            width_scale=width_scale,
        )
        self.stacked = stacked
        self._bar_data_list: list[_BarSetData] = []

    @property
    def stacked(self) -> bool:
        """Whether datasets are stacked rather than grouped."""
        return self._stacked

    @stacked.setter
    @deferred_axis_update
    def stacked(self, value: bool) -> None:
        self._stacked = bool(value)

    def add_dataset(
        self,
        heights: dict[str, float] | list[float] | pd.Series | pd.DataFrame,
        name: str | None = None,
        *,
        labels: list[str] | None = None,
        column: str | None = None,
        options: BarPlotOptions | None = None,
        facecolor: Color | None | Unset = UNSET,
        facealpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgealpha: float | None = None,
        edgewidth: float | None = None,
        add_extra_labels: bool = False,
        zorder: int | None = None,
    ) -> None:
        """Add one bar dataset (one bar per category) from precomputed heights.

        Args:
            heights (dict[str, float] | list[float] | pd.Series | pd.DataFrame): Bar heights
                by category. Dicts and Series map labels to heights; a DataFrame uses its
                index as labels (single column, or pass ``column=...``); a plain list is
                paired with ``labels`` (or the existing category labels).
            name (str | None, optional): Legend label for the dataset. Defaults to None.
            labels (list[str] | None, optional): Labels for list-based input. Defaults to None.
            column (str | None, optional): DataFrame column to read when ``heights`` is a
                DataFrame with more than one column. Defaults to None.
            options (BarPlotOptions | None, optional): Base styling whose values are used
                for any styling argument left unset. Defaults to None.
            facecolor (Color | None, optional): Bar fill color. Pass ``None`` for an
                unfilled bar. Omit to use the ``options`` default ``"default_grey"``.
            facealpha (float | None, optional): Bar fill alpha override. Defaults to None.
            edgecolor (Color | None, optional): Bar edge color. Pass ``None`` for no edge.
                Omit to use the ``options`` default ``"black"``.
            edgealpha (float | None, optional): Bar edge alpha override. Defaults to None.
            edgewidth (float, optional): Bar edge width. Defaults to ``0.8``.
            add_extra_labels (bool, optional): Whether to merge unseen incoming labels into
                existing category labels. Defaults to False.
            zorder (int, optional): Draw order for the dataset. Defaults to ``1``.

        Returns:
            None
        """
        # Bar heights are the same shape as pointset input (one float per category), so
        # the pointset converter handles dict/Series/DataFrame/list parsing.
        heights_dict = self._convert_pointset_to_dict(heights, labels, column=column)
        self._add_bar_set(
            heights_dict,
            name=name,
            options=options,
            facecolor=facecolor,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            add_extra_labels=add_extra_labels,
            zorder=zorder,
        )

    def add_counts_dataset(
        self,
        values: NumericArrayLike,
        name: str | None = None,
        *,
        column: str | None = None,
        options: BarPlotOptions | None = None,
        facecolor: Color | None | Unset = UNSET,
        facealpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgealpha: float | None = None,
        edgewidth: float | None = None,
        add_extra_labels: bool = False,
        zorder: int | None = None,
    ) -> None:
        """Add one bar dataset by counting occurrences of each distinct raw value.

        Each distinct value becomes a category (sorted ascending, labeled with its
        compact string form, e.g. ``3`` rather than ``3.0``) whose bar height is the
        number of times it appears. Values that never appear get no category; pass
        explicit heights to :meth:`add_dataset` to show empty categories.

        Args:
            values (NumericArrayLike): Raw values to count. Supported forms include
                iterables, numpy arrays, pandas Series, and pandas DataFrames.
            name (str | None, optional): Legend label for the dataset. Defaults to None.
            column (str | None, optional): The column name to use if values is a DataFrame.
            options (BarPlotOptions | None, optional): Base styling whose values are used
                for any styling argument left unset. Defaults to None.
            facecolor (Color | None, optional): Bar fill color. Pass ``None`` for an
                unfilled bar. Omit to use the ``options`` default ``"default_grey"``.
            facealpha (float | None, optional): Bar fill alpha override. Defaults to None.
            edgecolor (Color | None, optional): Bar edge color. Pass ``None`` for no edge.
                Omit to use the ``options`` default ``"black"``.
            edgealpha (float | None, optional): Bar edge alpha override. Defaults to None.
            edgewidth (float, optional): Bar edge width. Defaults to ``0.8``.
            add_extra_labels (bool, optional): Whether to merge unseen incoming labels into
                existing category labels. Defaults to False.
            zorder (int, optional): Draw order for the dataset. Defaults to ``1``.

        Returns:
            None

        Raises:
            ValueError: If ``values`` has no finite entries.
        """
        vals = _coerce_to_1d_finite_float_array(values, column=column, field="values")
        if vals.size == 0:
            raise ValueError("values: must have at least one finite entry.")

        uniques, counts = np.unique(vals, return_counts=True)
        # Default :g formatting can merge values beyond six significant digits.
        heights_dict = {
            np.format_float_positional(unique_value, trim="-"): float(count)
            for unique_value, count in zip(uniques, counts, strict=True)
        }
        self._add_bar_set(
            heights_dict,
            name=name,
            options=options,
            facecolor=facecolor,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            add_extra_labels=add_extra_labels,
            zorder=zorder,
        )

    def _add_bar_set(
        self,
        heights_dict: dict[str, float],
        *,
        name: str | None,
        options: BarPlotOptions | None,
        facecolor: Color | None | Unset,
        facealpha: float | None,
        edgecolor: Color | None | Unset,
        edgealpha: float | None,
        edgewidth: float | None,
        add_extra_labels: bool,
        zorder: int | None,
    ) -> None:
        """Resolve styling, sync labels, and store one bar dataset.

        Raises:
            ValueError: If ``heights_dict`` is empty.
        """
        if len(heights_dict) == 0:
            raise ValueError("heights is empty; provide a height for at least one category.")

        base = options if options is not None else BarPlotOptions()
        style = base.merged(
            facecolor=facecolor,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            zorder=zorder,
        )

        set_name = name or f"Set {len(self._bar_data_list) + 1}"
        bar_data = _BarSetData(name=set_name, heights_dict=heights_dict, style=style)
        self._sync_labels(
            list(heights_dict.keys()),
            add_extra_labels=add_extra_labels,
            item_name="bar set",
        )
        self._bar_data_list.append(bar_data)
        self._claim_legend_if_named(name)

    @property
    def _datasets(self) -> Sequence[object]:
        return self._bar_data_list

    def _draw_datasets(self) -> None:
        """Draw all bar datasets, side by side within each group or stacked."""
        assert self._labels is not None, (
            "Internal error: _labels should have been set by _sync_labels when adding bar data."
        )
        if self.stacked:
            # Every set contributes at every category (zero when a label is absent) so the
            # running bottoms stay aligned across sets.
            centers = self._category_centers
            bar_bottoms = np.zeros(len(centers))
            bar_width = self.group_width * self.width_scale
            for bar_data in self._bar_data_list:
                set_heights = np.array(
                    [bar_data.heights_dict.get(label, 0.0) for label in self._labels],
                    dtype=float,
                )
                self._draw_one_bar_set(bar_data, centers, set_heights, bar_width, bar_bottoms)
                bar_bottoms += set_heights
            return

        centers, offsets, bar_width = self._grouped_layout(len(self._bar_data_list))
        for k, bar_data in enumerate(self._bar_data_list):
            positions: list[float] = []
            heights: list[float] = []
            for _label, height, x in self._present_positions(
                bar_data.heights_dict, centers + offsets[k]
            ):
                positions.append(x)
                heights.append(height)

            if len(positions) == 0:
                continue
            self._draw_one_bar_set(
                bar_data,
                np.asarray(positions),
                np.asarray(heights),
                bar_width,
                np.zeros(len(positions)),
            )

    def _draw_one_bar_set(
        self,
        bar_data: _BarSetData,
        positions: np.ndarray,
        heights: np.ndarray,
        bar_width: float,
        bottoms: np.ndarray,
    ) -> None:
        """Draw and track the bars for one dataset.

        Args:
            bar_data (_BarSetData): The dataset's heights and styling.
            positions (np.ndarray): Bar center x positions.
            heights (np.ndarray): Bar heights aligned with ``positions``.
            bar_width (float): Width of each bar.
            bottoms (np.ndarray): Bar base y positions aligned with ``positions``.

        Returns:
            None
        """
        style = bar_data.style
        bar_container = self._ax.bar(
            positions,
            heights,
            width=bar_width,
            bottom=bottoms,
            align="center",
            facecolor=self._resolved_rgba(style.facecolor, style.facealpha, field="facecolor"),
            edgecolor=self._resolved_rgba(style.edgecolor, style.edgealpha, field="edgecolor"),
            linewidth=style.edgewidth,
            zorder=style.zorder,
        )
        # BarContainer is iterable over its Rectangle patches.
        self._artists.track(bar_container)

    def _dataset_legend_handles(self) -> list[LegendHandle]:
        """Legend handles for the bar datasets."""
        return self._patch_legend_handles(
            (bar_data.name, bar_data.style) for bar_data in self._bar_data_list
        )
