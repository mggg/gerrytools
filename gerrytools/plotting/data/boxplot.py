from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence, SupportsFloat, cast

import matplotlib.cbook as cbook
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from gerrytools.logging import get_logger
from gerrytools.plotting.data._categorical_distribution_base import CategoricalDistributionPlotBase
from gerrytools.plotting.data.options import BoxPlotOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.plotting.utils import UNSET, Unset
from gerrytools.typing import Color, LegendHandle

logger = get_logger(__name__)


# Field names a summary-stats mapping/DataFrame must provide for each category.
_REQUIRED_STAT_FIELDS: tuple[str, ...] = (
    "median",
    "lower_quartile",
    "upper_quartile",
    "lower_whisker",
    "upper_whisker",
)


def _as_float(value: object) -> float:
    """Coerce a scalar (Python or NumPy) value to a plain ``float``."""
    return float(cast("SupportsFloat | str", value))


def _coerce_fliers(value: object) -> tuple[float, ...]:
    """Coerce a fliers input into a tuple of floats.

    Accepts a sequence or NumPy array of values. ``None`` and lone scalars (for
    example a NaN produced by an absent DataFrame cell) are treated as "no
    fliers" so that summary-stats input without outliers parses cleanly.

    Args:
        value (object): Raw fliers value from a mapping or DataFrame cell.

    Returns:
        tuple[float, ...]: Outlier values, possibly empty.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("fliers must be a sequence of numbers, not a string.")
    if isinstance(value, np.ndarray):
        return tuple(_as_float(flier) for flier in cast("Iterable[object]", value))
    if isinstance(value, Sequence):
        return tuple(_as_float(flier) for flier in value)
    # A lone scalar (e.g. NaN from a missing DataFrame cell) means "no fliers".
    return ()


@dataclass(frozen=True)
class BoxPlotStats:
    """Precomputed summary statistics for a single box.

    Pass these to :meth:`BoxPlot.add_stats_dataset` to draw a box from
    an already-computed five-number summary instead of from raw samples — useful
    when the underlying ensemble is too large to keep in memory or the statistics
    were computed elsewhere.

    Attributes:
        median (float): Median (Q2); the line drawn inside the box.
        lower_quartile (float): Lower quartile (Q1); the bottom of the box.
        upper_quartile (float): Upper quartile (Q3); the top of the box.
        lower_whisker (float): Lower whisker cap position.
        upper_whisker (float): Upper whisker cap position.
        mean (float | None): Optional mean. Rendered only when every box in the
            dataset provides one. Defaults to None.
        fliers (Sequence[float]): Optional outlier values, rendered when
            ``showfliers`` is enabled for the dataset. Defaults to empty.

    Raises:
        ValueError: If any required statistic is non-finite or the values do not
            satisfy ``lower_whisker <= lower_quartile <= median <= upper_quartile
            <= upper_whisker``.
    """

    median: float
    lower_quartile: float
    upper_quartile: float
    lower_whisker: float
    upper_whisker: float
    mean: float | None = None
    fliers: Sequence[float] = ()

    def __post_init__(self) -> None:
        median = _as_float(self.median)
        lower_quartile = _as_float(self.lower_quartile)
        upper_quartile = _as_float(self.upper_quartile)
        lower_whisker = _as_float(self.lower_whisker)
        upper_whisker = _as_float(self.upper_whisker)

        for field_name, field_value in (
            ("median", median),
            ("lower_quartile", lower_quartile),
            ("upper_quartile", upper_quartile),
            ("lower_whisker", lower_whisker),
            ("upper_whisker", upper_whisker),
        ):
            if not math.isfinite(field_value):
                raise ValueError(f"{field_name} must be a finite number; got {field_value!r}.")

        if not (lower_whisker <= lower_quartile <= median <= upper_quartile <= upper_whisker):
            raise ValueError(
                "Box plot statistics must satisfy lower_whisker <= lower_quartile <= median "
                "<= upper_quartile <= upper_whisker; got "
                f"lower_whisker={lower_whisker}, lower_quartile={lower_quartile}, "
                f"median={median}, upper_quartile={upper_quartile}, upper_whisker={upper_whisker}."
            )

        object.__setattr__(self, "median", median)
        object.__setattr__(self, "lower_quartile", lower_quartile)
        object.__setattr__(self, "upper_quartile", upper_quartile)
        object.__setattr__(self, "lower_whisker", lower_whisker)
        object.__setattr__(self, "upper_whisker", upper_whisker)

        if self.mean is not None:
            mean = _as_float(self.mean)
            if not math.isfinite(mean):
                raise ValueError(f"mean must be a finite number when provided; got {mean!r}.")
            object.__setattr__(self, "mean", mean)

        object.__setattr__(self, "fliers", tuple(_as_float(flier) for flier in self.fliers))

    @staticmethod
    def from_samples(
        values: Sequence[float] | NDArray,
        *,
        percentiles: tuple[float, float] = (1, 99),
    ) -> BoxPlotStats:
        """Reduce raw samples to box statistics, matching Matplotlib's numbers.

        Uses ``matplotlib.cbook.boxplot_stats`` (what ``Axes.boxplot`` runs internally)
        so a box drawn from these statistics is identical to one drawn from the raw
        samples. Non-finite samples are dropped first. The mean is left unset so
        raw-sample datasets render without mean markers.

        Args:
            values (Sequence[float] | NDArray): Raw sample values.
            percentiles (tuple[float, float], optional): Lower/upper whisker
                percentiles. Defaults to ``(1, 99)``.

        Returns:
            BoxPlotStats: The reduced statistics.

        Raises:
            ValueError: If ``values`` has no finite entries.
        """
        finite = np.asarray(values, dtype=float).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("values must have at least one finite entry.")

        (stats,) = cbook.boxplot_stats(finite, whis=percentiles)
        return BoxPlotStats(
            median=stats["med"],
            lower_quartile=stats["q1"],
            upper_quartile=stats["q3"],
            lower_whisker=stats["whislo"],
            upper_whisker=stats["whishi"],
            fliers=tuple(float(flier) for flier in stats["fliers"]),
        )

    def to_bxp_dict(self, label: str) -> dict[str, object]:
        """Convert to the per-box dict consumed by Matplotlib's ``Axes.bxp``.

        Args:
            label (str): Category label for the box.

        Returns:
            dict[str, object]: Matplotlib ``bxp`` statistics dict. The ``mean``
            key is included only when :attr:`mean` is set.
        """
        stats: dict[str, object] = {
            "label": label,
            "med": self.median,
            "q1": self.lower_quartile,
            "q3": self.upper_quartile,
            "whislo": self.lower_whisker,
            "whishi": self.upper_whisker,
            "fliers": list(self.fliers),
        }
        if self.mean is not None:
            stats["mean"] = self.mean
        return stats


@dataclass(frozen=True)
class _BoxPlotSetData:
    """One boxplot dataset: per-category statistics plus styling.

    Raw-sample input is reduced to :class:`BoxPlotStats` at add time, so every
    dataset draws through the same ``Axes.bxp`` path.
    """

    name: str
    stats_dict: dict[str, BoxPlotStats]
    style: BoxPlotOptions


class BoxPlot(CategoricalDistributionPlotBase):
    """Create grouped boxplot comparison figures across categories.

    Add data either from raw samples via :meth:`add_dataset` or from a
    precomputed five-number summary via :meth:`add_stats_dataset`. Both
    kinds of sets can be combined on a single figure and share the grouped layout.
    """

    _dataset_noun = "boxplot set"

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
    ) -> None:
        """Initialize a BoxPlot.

        Toggle the per-group vertical guide lines with
        :meth:`display_group_separators` after construction.

        Args:
            figure_size (tuple[float, float] | None, optional): Figure size in inches. Defaults to
                Matplotlib's setting.
            dpi (int | None, optional): Figure resolution. Defaults to Matplotlib's setting.
            ax (Axes | None, optional): Existing axes to draw on, or None to create them. Defaults
                to None.
            legend (bool | None, optional): Whether to draw a legend. None leaves existing axes
                state unchanged. Defaults to None.
            xlabel (str | None, optional): X-axis label. Defaults to None.
            ylabel (str | None, optional): Y-axis label. Defaults to None.
            title (str | None, optional): Axes title. Defaults to None.
            width_scale (float, optional): Relative width of boxes within each slot. Defaults to
                0.8.
            group_width (float, optional): Width allocated to each category. Defaults to 0.7.
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
        self._boxplot_data_list: list[_BoxPlotSetData] = []

    @staticmethod
    def _sync_default_flier_zorder(style: BoxPlotOptions) -> BoxPlotOptions:
        """Layer fliers with the rest of the set when no explicit flier styling was given.

        An explicit ``flier_options`` (or ``options``) keeps its own zorder.
        """
        return style.merged(flier_options=replace(style.flier_options, zorder=style.zorder))

    def _store_dataset(
        self,
        stats_dict: dict[str, BoxPlotStats],
        *,
        name: str | None,
        style: BoxPlotOptions,
        labels: list[str],
        add_extra_labels: bool,
        item_name: str,
    ) -> None:
        """Sync labels and store one dataset.

        ``labels`` may include categories absent from ``stats_dict`` (e.g. empty raw-sample
        categories): they still claim a slot on the axis, but draw no box.
        """
        self._sync_labels(labels, add_extra_labels=add_extra_labels, item_name=item_name)
        set_name = name or f"Set {len(self._boxplot_data_list) + 1}"
        style = replace(style, flier_options=replace(style.flier_options))
        self._boxplot_data_list.append(
            _BoxPlotSetData(name=set_name, stats_dict=stats_dict, style=style)
        )
        self._claim_legend_if_named(name)

    def add_dataset(
        self,
        scores: (
            Mapping[str, Sequence[float]]
            | Sequence[float]
            | Sequence[Sequence[float]]
            | pd.DataFrame
            | NDArray
        ),
        name: str | None = None,
        *,
        category_labels: list[str] | None = None,
        options: BoxPlotOptions | None = None,
        facecolor: Color | None | Unset = UNSET,
        facealpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgealpha: float | None = None,
        edgewidth: float | None = None,
        percentiles: tuple[float, float] | None = None,
        showfliers: bool | None = None,
        flier_options: PointMarkerOptions | None = None,
        add_extra_labels: bool = False,
        zorder: int | None = None,
    ) -> None:
        """Add one boxplot dataset (one box per category) to the figure.

        The raw samples are reduced to per-category :class:`BoxPlotStats` immediately
        (matching Matplotlib's own statistics), so only the summary is retained. Non-finite
        samples are dropped per category; a category with no finite samples keeps its axis
        slot but draws no box.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                Distribution input by category.
            category_labels (list[str] | None, optional): Labels for list-based input.
                Defaults to None.
            name (str | None, optional): Legend label for the dataset. Defaults to None.
            options (BoxPlotOptions | None, optional): Base styling whose values are used
                for any styling argument left unset. Defaults to None.
            facecolor (Color | None, optional): Box fill color. Pass ``None`` for an
                unfilled (transparent) box. Omit to use the ``options`` default ``"default_grey"``.
            facealpha (float | None, optional): Box fill alpha override. Defaults to None.
            edgecolor (Color | None, optional): Box edge color. Pass ``None`` for no edge.
                Omit to use the ``options`` default ``"black"``.
            edgealpha (float | None, optional): Box edge alpha override. Defaults to None.
            edgewidth (float, optional): Box edge width. Defaults to ``0.8``.
            percentiles (tuple[float, float], optional): Lower/upper whisker percentiles.
                Defaults to ``(1, 99)``.
            showfliers (bool, optional): Whether to show outlier markers. Defaults to False.
            flier_options (PointMarkerOptions | None, optional): Outlier marker options.
                Defaults to None.
            add_extra_labels (bool, optional): Whether to merge unseen incoming labels into
                existing category labels. Defaults to False.
            zorder (int, optional): Draw order for the dataset. Defaults to ``1``.

        Returns:
            None

        Raises:
            ValueError: If ``scores`` is empty.
        """
        style = (options if options is not None else BoxPlotOptions()).merged(
            facecolor=facecolor,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            percentiles=percentiles,
            showfliers=showfliers,
            flier_options=flier_options,
            zorder=zorder,
        )
        if options is None and flier_options is None:
            style = self._sync_default_flier_zorder(style)

        scores_dict = self._convert_distribution_data_to_dictionary(scores, category_labels)
        if len(scores_dict) == 0:
            raise ValueError("scores is empty; provide scores for at least one category.")

        # Filter non-finite samples before the emptiness check so a NaN-only category is
        # skipped (keeping its label slot) for every input container, not just DataFrames.
        stats_dict: dict[str, BoxPlotStats] = {}
        for label, vals in scores_dict.items():
            finite_vals = np.asarray(vals, dtype=float)
            finite_vals = finite_vals[np.isfinite(finite_vals)]
            if finite_vals.size > 0:
                stats_dict[label] = BoxPlotStats.from_samples(
                    finite_vals, percentiles=style.percentiles
                )
        self._store_dataset(
            stats_dict,
            name=name,
            style=style,
            labels=list(scores_dict.keys()),
            add_extra_labels=add_extra_labels,
            item_name="boxplot set",
        )

    @staticmethod
    def _box_plot_stats_from_mapping(
        values: Mapping[str, object],
        *,
        category: str,
    ) -> BoxPlotStats:
        """Build a :class:`BoxPlotStats` from a mapping of field names to values.

        Args:
            values (Mapping[str, object]): Field-name to value mapping (e.g. a dict
                or a DataFrame row converted via ``Series.to_dict``).
            category (str): Category label, used in error messages.

        Returns:
            BoxPlotStats: The parsed statistics.

        Raises:
            ValueError: If any required field is missing.
        """
        missing = [name for name in _REQUIRED_STAT_FIELDS if name not in values]
        if missing:
            raise ValueError(
                f"Box plot stats for category {category!r} is missing required "
                f"field(s): {', '.join(missing)}."
            )

        raw_mean = values.get("mean")
        if raw_mean is None:
            mean: float | None = None
        else:
            mean_value = _as_float(raw_mean)
            # A NaN mean (e.g. an absent DataFrame cell) means "no mean".
            mean = None if math.isnan(mean_value) else mean_value

        return BoxPlotStats(
            median=_as_float(values["median"]),
            lower_quartile=_as_float(values["lower_quartile"]),
            upper_quartile=_as_float(values["upper_quartile"]),
            lower_whisker=_as_float(values["lower_whisker"]),
            upper_whisker=_as_float(values["upper_whisker"]),
            mean=mean,
            fliers=_coerce_fliers(values.get("fliers", ())),
        )

    @staticmethod
    def _convert_boxplot_stats_to_dictionary(
        stats: Mapping[str, BoxPlotStats | Mapping[str, object]] | pd.DataFrame,
    ) -> dict[str, BoxPlotStats]:
        """Normalize summary-stats input into a ``{category: BoxPlotStats}`` dict.

        Args:
            stats (Mapping[str, BoxPlotStats | Mapping[str, object]] | pd.DataFrame):
                Per-category statistics. A DataFrame uses its index as categories
                and columns as stat field names.

        Returns:
            dict[str, BoxPlotStats]: Category label to statistics mapping.

        Raises:
            TypeError: If ``stats`` (or one of its values) is an unsupported type.
        """
        if isinstance(stats, pd.DataFrame):
            return {
                str(category): BoxPlot._box_plot_stats_from_mapping(
                    cast("Mapping[str, object]", row.to_dict()),
                    category=str(category),
                )
                for category, row in stats.iterrows()
            }

        if isinstance(stats, Mapping):
            converted: dict[str, BoxPlotStats] = {}
            for category, value in stats.items():
                category_label = str(category)
                if isinstance(value, BoxPlotStats):
                    converted[category_label] = value
                elif isinstance(value, Mapping):
                    converted[category_label] = BoxPlot._box_plot_stats_from_mapping(
                        cast("Mapping[str, object]", value),
                        category=category_label,
                    )
                else:
                    raise TypeError(
                        f"Box plot stats for category {category_label!r} must be a BoxPlotStats "
                        f"or a mapping of field names to values; got {type(value).__name__}."
                    )
            return converted

        raise TypeError(
            "stats must be a Mapping[str, BoxPlotStats | Mapping[str, float]] or a pandas "
            "DataFrame."
        )

    def add_stats_dataset(
        self,
        stats: Mapping[str, BoxPlotStats | Mapping[str, object]] | pd.DataFrame,
        name: str | None = None,
        *,
        options: BoxPlotOptions | None = None,
        facecolor: Color | None | Unset = UNSET,
        facealpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgealpha: float | None = None,
        edgewidth: float | None = None,
        showfliers: bool | None = None,
        flier_options: PointMarkerOptions | None = None,
        add_extra_labels: bool = False,
        zorder: int | None = None,
    ) -> None:
        """Add one boxplot dataset from precomputed summary statistics.

        Use this instead of :meth:`add_dataset` when you already have a
        five-number summary per category (median, quartiles, whiskers) rather than
        the raw samples. Stats and raw datasets can be mixed on the same figure; all
        sets share the grouped layout and are positioned together.

        Args:
            stats (Mapping[str, BoxPlotStats | Mapping[str, object]] | pd.DataFrame):
                Per-category statistics. Each value may be a :class:`BoxPlotStats`,
                a mapping of field names (``median``, ``lower_quartile``,
                ``upper_quartile``, ``lower_whisker``, ``upper_whisker``, and the
                optional ``mean`` / ``fliers``) to values, or — for a DataFrame —
                a row whose columns are those field names (index = categories).
            name (str | None, optional): Legend label for the dataset. Defaults to None.
            options (BoxPlotOptions | None, optional): Base styling whose values are
                used for any styling argument left unset. The ``percentiles`` field
                is ignored because whiskers are supplied directly. Defaults to None.
            facecolor (Color | None, optional): Box fill color. Pass ``None`` for an
                unfilled (transparent) box. Omit to use the ``options`` default ``"default_grey"``.
            facealpha (float | None, optional): Box fill alpha override. Defaults to None.
            edgecolor (Color | None, optional): Box edge color. Pass ``None`` for no edge.
                Omit to use the ``options`` default ``"black"``.
            edgealpha (float | None, optional): Box edge alpha override. Defaults to None.
            edgewidth (float | None, optional): Box edge width. Defaults to ``0.8``.
            showfliers (bool | None, optional): Whether to render outlier markers from
                each box's ``fliers``. Defaults to False.
            flier_options (PointMarkerOptions | None, optional): Outlier marker options.
                Defaults to None.
            add_extra_labels (bool, optional): Whether to merge unseen incoming labels
                into existing category labels. Defaults to False.
            zorder (int | None, optional): Draw order for the dataset. Defaults to ``1``.

        Returns:
            None

        Raises:
            ValueError: If ``stats`` is empty.
        """
        style = (options if options is not None else BoxPlotOptions()).merged(
            facecolor=facecolor,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            showfliers=showfliers,
            flier_options=flier_options,
            zorder=zorder,
        )
        if options is None and flier_options is None:
            style = self._sync_default_flier_zorder(style)

        stats_dict = self._convert_boxplot_stats_to_dictionary(stats)
        if len(stats_dict) == 0:
            raise ValueError("stats is empty; provide statistics for at least one category.")

        self._store_dataset(
            stats_dict,
            name=name,
            style=style,
            labels=list(stats_dict.keys()),
            add_extra_labels=add_extra_labels,
            item_name="boxplot stats set",
        )

    @property
    def _datasets(self) -> Sequence[object]:
        return self._boxplot_data_list

    def _draw_datasets(self) -> None:
        """Draw all boxplot sets through ``Axes.bxp``."""
        centers, offsets, widths = self._grouped_layout(len(self._boxplot_data_list))

        for k, set_data in enumerate(self._boxplot_data_list):
            bxp_stats: list[dict[str, object]] = []
            pos_k: list[float] = []
            means_present: list[bool] = []
            for label, stat, x in self._present_positions(
                set_data.stats_dict, centers + offsets[k]
            ):
                bxp_stats.append(stat.to_bxp_dict(label))
                pos_k.append(x)
                means_present.append(stat.mean is not None)

            if len(bxp_stats) == 0:
                continue

            style = set_data.style
            drawn = self._ax.bxp(
                bxp_stats,
                positions=pos_k,
                widths=widths,
                patch_artist=True,
                manage_ticks=False,
                showfliers=style.showfliers,
                # Means render only when every drawn box in the set provides one.
                showmeans=all(means_present),
                # Cast to satisfy the type-checker: Matplotlib stubs expect
                # a plain ``dict[str, object]`` for ``flierprops``.
                flierprops=cast(
                    dict[str, object],
                    style.flier_options.to_mpl_settings_dict(),
                ),
            )
            self._style_box_set(drawn, style)

        assert self._labels is not None
        self._ax.set_xlim(0.5, len(self._labels) + 0.5)

    def _style_box_set(self, drawn: dict[str, Any], style: BoxPlotOptions) -> None:
        """Apply gerrytools styling to the artists of one drawn box set.

        Args:
            drawn (dict[str, Any]): Matplotlib artist dict from ``Axes.bxp``.
            style (BoxPlotOptions): The set's styling.

        Returns:
            None
        """
        for key in ("boxes", "whiskers", "caps", "medians", "means", "fliers"):
            for artist in drawn.get(key, []):
                self._artists.track(artist)

        facecolor = self._resolved_rgba(style.facecolor, style.facealpha, field="facecolor")
        edgecolor = self._resolved_rgba(style.edgecolor, style.edgealpha, field="edgecolor")

        for patch in drawn["boxes"]:
            patch.set_facecolor(facecolor)
            patch.set_linewidth(style.edgewidth)
            patch.set_edgecolor(edgecolor)
            patch.set_zorder(style.zorder)

        for key in ("whiskers", "caps", "medians", "means"):
            for artist in drawn.get(key, []):
                artist.set_color(edgecolor)
                artist.set_linewidth(style.edgewidth)
                artist.set_zorder(style.zorder)

        # Fliers keep the zorder from flier_options (already applied via flierprops).

    def _dataset_legend_handles(self) -> list[LegendHandle]:
        """Legend handles for the boxplot sets."""
        return self._patch_legend_handles(
            (boxplot_data.name, boxplot_data.style) for boxplot_data in self._boxplot_data_list
        )
