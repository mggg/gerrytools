from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from numpy.random import Generator

from gerrytools.logging import get_logger
from gerrytools.plotting._axes_backed import deferred_axis_update
from gerrytools.plotting._rng import resolve_numpy_rng, spawn_child_seeds
from gerrytools.plotting.data._categorical_distribution_base import CategoricalDistributionPlotBase
from gerrytools.plotting.data.options import SeaLevelLineOptions
from gerrytools.plotting.mpl.marker_options import PointMarkerOptions
from gerrytools.plotting.utils import UNSET, Unset, _replace_with_color_overrides
from gerrytools.typing import CategoryKey, Color, LegendHandle

logger = get_logger(__name__)


@dataclass(frozen=True)
class _SeaLevelSetData:
    """One connected line of per-category scores; styling validates in the options classes."""

    name: str
    scores_dict: dict[str, float]
    style: SeaLevelLineOptions = field(default_factory=SeaLevelLineOptions)
    markersettings: PointMarkerOptions = field(default_factory=PointMarkerOptions)


class SeaLevelPlot(CategoricalDistributionPlotBase):
    """Connected per-category score lines ("sea levels") across categorical x positions."""

    _dataset_noun = "sealevel set"

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
        jitter_rng_seed: int | None = None,
        jitter_rng: Generator | None = None,
    ) -> None:
        """Initialize a SeaLevelPlot instance.

        Args:
            figure_size (tuple[float, float] | None, optional): The size of the
                figure in inches. Defaults to ``(10, 6)`` when ``ax`` is not provided.
            dpi (int | None, optional): The dots per inch (DPI) of the figure.
                Defaults to ``300`` when ``ax`` is not provided.
            ax (matplotlib.axes.Axes | None, optional): Render onto an existing
                matplotlib ``Axes`` instead of creating a fresh figure. Defaults to None.
            legend (bool | None, optional): Whether to include a legend in the plot.
                ``None`` selects the class default (False). Defaults to None.
            xlabel (str | None, optional): The label for the x-axis. Defaults to None.
            ylabel (str | None, optional): The label for the y-axis. Defaults to None.
            title (str | None, optional): The title of the plot. Defaults to None.
            jitter_rng_seed (int | None, optional): Seed for reproducible jitter placement.
                Defaults to None.
            jitter_rng (Generator | None, optional): Explicit NumPy generator to use for
                jitter instead of constructing one from ``jitter_rng_seed``. Defaults to None.

        Toggle the grid with :meth:`display_grid` after construction.
        """
        super().__init__(
            figure_size=figure_size,
            dpi=dpi,
            ax=ax,
            legend=legend,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            group_width=0.7,
            width_scale=0.8,
        )

        self._sealevel_data_list: list[_SeaLevelSetData] = []
        self._maximum_vertical_jitter_per_category: float | dict[str, float] = {}
        self._maximum_horizontal_jitter_per_category: float | dict[str, float] = {}

        self._jitter_rng, self._jitter_rng_seed = resolve_numpy_rng(
            seed=jitter_rng_seed,
            rng=jitter_rng,
            field_name="jitter_rng_seed",
        )
        self._jitter_base_seed = self._derive_jitter_base_seed()

    def _derive_jitter_base_seed(self) -> int:
        """One per-plot base seed so every rebuild replays the same jitter stream.

        With an explicit seed the stream is fully reproducible across processes; with a
        user-supplied generator (or neither) the base seed is drawn once from it, so
        rebuilds of this plot stay identical while distinct plots still differ.
        """
        if self._jitter_rng_seed is not None:
            return self._jitter_rng_seed
        return spawn_child_seeds(self._jitter_rng, 1)[0]

    @property
    def jitter_rng_seed(self) -> int | None:
        """Get the RNG seed used for deterministic jitter placement.

        Returns:
            int | None: Current jitter RNG seed, or None for nondeterministic behavior.
        """
        return self._jitter_rng_seed

    @jitter_rng_seed.setter
    @deferred_axis_update
    def jitter_rng_seed(self, seed: int | None) -> None:
        """Set the RNG seed used for deterministic jitter placement.

        Args:
            seed (int | None): Integer seed value, or None to use nondeterministic randomness.

        Returns:
            None

        Raises:
            TypeError: If ``seed`` is neither ``int`` nor ``None``.
        """
        self._jitter_rng, self._jitter_rng_seed = resolve_numpy_rng(
            seed=seed,
            field_name="jitter_rng_seed",
        )
        self._jitter_base_seed = self._derive_jitter_base_seed()

    def _validate_jitter_per_category(self, jitter: dict[str, float]) -> None:
        """Validate a per-category jitter dictionary against the current labels."""
        if not isinstance(jitter, dict):
            raise TypeError("jitter must be a dictionary mapping labels to floats.")
        if not all(isinstance(v, Real) for v in jitter.values()):
            raise TypeError("All values in jitter must be real numbers.")
        if not all(v >= 0 for v in jitter.values()):
            raise ValueError("All jitter values must be nonnegative.")
        if not all(math.isfinite(float(v)) for v in jitter.values()):
            raise ValueError("All jitter values must be finite.")
        if len(self._labels or []) == 0:
            raise ValueError("No labels defined yet; cannot set jitter per category.")
        if not set(jitter.keys()).issubset(set(self._labels or [])):
            extra_keys = set(jitter.keys()) - set(self._labels or [])
            raise ValueError(
                f"All keys in jitter must be among the existing labels. Extra keys: {extra_keys}"
            )

    def _coerce_jitter(self, jitter: float | dict[str, float]) -> float | dict[str, float]:
        """Validate and copy a uniform-or-per-category jitter input.

        - ``dict[str, float]`` is validated and returned as-is.
        - A scalar float applies to current and future labels.
        """
        if isinstance(jitter, dict):
            self._validate_jitter_per_category(jitter)
            # Do not let later caller mutation change rendered positions.
            return dict(jitter)
        if not isinstance(jitter, Real):
            raise TypeError(
                "jitter must be a float or a dictionary mapping labels to floats; "
                f"got {type(jitter).__name__!r}."
            )
        if len(self._labels or []) == 0:
            raise ValueError("No labels defined yet; cannot set jitter.")
        jit = float(jitter)
        if not math.isfinite(jit) or jit < 0:
            raise ValueError("jitter must be a finite nonnegative number.")
        return jit

    @deferred_axis_update
    def set_vertical_jitter(self, jitter: float | dict[str, float]) -> None:
        """Set the maximum vertical jitter applied to category points.

        Pass a single ``float`` to apply the same jitter uniformly to every
        category, or pass a ``dict`` mapping label to float for per-category
        jitter. Categories not in the dict get no jitter.

        Any point in a given category is jittered randomly within
        ``[-jitter, +jitter]``.

        Args:
            jitter (float | dict[str, float]): Uniform jitter value, or
                per-category mapping.
        """
        self._maximum_vertical_jitter_per_category = self._coerce_jitter(jitter)

    @deferred_axis_update
    def set_horizontal_jitter(self, jitter: float | dict[str, float]) -> None:
        """Set the maximum horizontal jitter applied to category points.

        Pass a single ``float`` to apply the same jitter uniformly to every
        category, or pass a ``dict`` mapping label to float for per-category
        jitter. Categories not in the dict get no jitter.

        Args:
            jitter (float | dict[str, float]): Uniform jitter value, or
                per-category mapping.
        """
        self._maximum_horizontal_jitter_per_category = self._coerce_jitter(jitter)

    def _convert_score_data_to_dictionary(
        self,
        scores: dict[str, int | float] | list[int | float] | pd.Series | pd.DataFrame,
        category_labels: list[str] | None = None,
        df_row_index: CategoryKey | None = None,
    ) -> dict[str, float]:
        """Convert supported score inputs into a label-to-value dictionary.

        DataFrame input selects one *row* via ``df_row_index`` (each column is a category);
        every other input converts via the shared point-set conversion.

        Args:
            scores (dict[str, int | float] | list[int | float] | pd.Series | pd.DataFrame):
                Input scores. Lists require ``category_labels``; DataFrames require
                ``df_row_index``.
            category_labels (list[str] | None, optional): Labels for list input. Defaults to None.
            df_row_index (CategoryKey | None, optional): Row selector for DataFrame input.
                Defaults to None.

        Returns:
            dict[str, float]: Mapping from category label to numeric value.

        Raises:
            ValueError: If conversion fails, inputs are empty, or values are non-finite.
            TypeError: If ``scores`` uses an unsupported input type.
        """
        if isinstance(scores, pd.DataFrame):
            if df_row_index is None:
                raise ValueError(
                    "If scores is a DataFrame, df_row_index must be provided to select the row."
                )
            if df_row_index not in scores.index:
                raise ValueError(f"df_row_index {df_row_index} not found in DataFrame index.")
            row = scores.loc[df_row_index]
            if isinstance(row, pd.DataFrame):
                raise ValueError(
                    "The selected df_row_index corresponds to multiple rows. "
                    "Please provide a df_row_index that selects a single row."
                )
            out_dict = {str(idx): float(value) for idx, value in row.items()}
        else:
            # Without explicit labels, list input falls back to the plot's existing labels
            # (matching the other categorical plots); only a label-less plot must raise.
            if isinstance(scores, list) and category_labels is None and self._labels is None:
                raise ValueError(
                    "If scores is a list, category_labels must be provided to map values to labels."
                )
            out_dict = self._convert_pointset_to_dict(scores, category_labels)

        if not out_dict:
            raise ValueError(
                "Could not convert scores to dictionary. Please check that the input is not empty."
            )
        if any(not math.isfinite(v) for v in out_dict.values()):
            raise ValueError("All score values must be finite numbers.")
        return out_dict

    def add_dataset(
        self,
        scores: dict[str, float] | list[float] | pd.Series | pd.DataFrame,
        name: str | None = None,
        *,
        category_labels: list[str] | None = None,
        df_row_index: CategoryKey | None = None,
        line_options: SeaLevelLineOptions | None = None,
        marker_options: PointMarkerOptions | None = None,
        linecolor: Color | None | Unset = UNSET,
        linealpha: float | None = None,
        linewidth: float | None = None,
        linestyle: str | None = None,
        marker: str | None = None,
        markerfacecolor: Color | None | Unset = UNSET,
        markerfacealpha: float | None = None,
        markersize: float | None = None,
        markeredgecolor: Color | None | Unset = UNSET,
        markeredgealpha: float | None = None,
        markeredgewidth: float | None = None,
        zorder: int | None = None,
        add_extra_labels: bool = False,
    ) -> None:
        """Add a set of points to the figure.

        Args:
            scores (dict[str, float] | list[float] | pd.Series | pd.DataFrame):
                The pointset values. Can be a dictionary mapping labels to values,
                a list of values, a Series, or a DataFrame.
            category_labels (list[str] | None, optional): The labels corresponding to the
                scores list, if scores is provided as a list. When omitted, list input falls
                back to the plot's existing category labels. Ignored if scores is a dict,
                Series, or DataFrame. Defaults to None.
            df_row_index (CategoryKey | None, optional): The row index to select if scores is a
                DataFrame.
                Defaults to None.
            name (str | None, optional): The name of the point set for the legend.
                Defaults to None.
            line_options (SeaLevelLineOptions | None, optional): Pre-built line styling. Any
                line styling kwarg passed explicitly overrides the corresponding field.
                Defaults to None.
            marker_options (PointMarkerOptions | None, optional): Pre-built marker styling. Any
                marker styling kwarg passed explicitly overrides the corresponding field. When
                None, markers inherit the resolved line color. Defaults to None.
            linecolor (Color | None, optional): The color of the line connecting the points. Pass
                ``None`` for no line. Defaults to "black".
            linealpha (float | None, optional): The alpha transparency of the line.
                Defaults to None.
            linewidth (float, optional): The width of the line. Defaults to 1.5.
            linestyle (str, optional): The style of the line. Defaults to "-".
            marker (str, optional): The marker style for the points. Defaults to "o".
            markerfacecolor (Color | None, optional): The face color of the markers. Pass ``None``
                for no fill. When omitted, uses ``linecolor``.
            markerfacealpha (float | None, optional): The alpha transparency of the marker face.
                Defaults to None, which uses linealpha.
            markersize (float, optional): The size of the markers. Defaults to 7.0.
            markeredgecolor (Color | None, optional): The edge color of the markers. Pass ``None``
                for no edge. When omitted, uses ``markerfacecolor``.
            markeredgealpha (float | None, optional): The alpha transparency of the marker edge.
                Defaults to None, which uses markerfacealpha if defined and linealpha otherwise.
            markeredgewidth (float, optional): The width of the marker edge. Defaults to 0.8.
            zorder (int, optional): The z-order for layering the plot elements.
                Defaults to 2.
            add_extra_labels (bool, optional): Whether to merge unseen incoming labels into
                existing category labels. Defaults to False.

        Note:
            Markers ride the connecting ``Line2D``, so they always draw at the line's
            ``zorder``; a ``zorder`` set on ``marker_options`` is ignored.

        Returns:
            None

        """
        scores_dict = self._convert_score_data_to_dictionary(scores, category_labels, df_row_index)

        line_base = line_options if line_options is not None else SeaLevelLineOptions()
        line_style = _replace_with_color_overrides(
            line_base,
            ("linecolor", "linealpha"),
            linecolor=linecolor,
            linealpha=linealpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )

        # When no marker options are given, markers inherit the resolved line color: the edge
        # follows the (possibly kwarg-overridden) face; a user-supplied marker_options is
        # honored verbatim, so an explicit black edge stays black.
        default_markerfacecolor = (
            line_style.linecolor if isinstance(markerfacecolor, Unset) else markerfacecolor
        )
        default_markerfacealpha = (
            line_style.linealpha if markerfacealpha is None else markerfacealpha
        )
        default_markeredgecolor = (
            default_markerfacecolor if isinstance(markeredgecolor, Unset) else markeredgecolor
        )
        default_markeredgealpha = (
            default_markerfacealpha if markeredgealpha is None else markeredgealpha
        )
        marker_base = (
            marker_options
            if marker_options is not None
            else PointMarkerOptions(
                markerfacecolor=default_markerfacecolor,
                markerfacealpha=default_markerfacealpha,
                marker="o",
                markersize=7.0,
                markeredgecolor=default_markeredgecolor,
                markeredgealpha=default_markeredgealpha,
                markeredgewidth=0.8,
            )
        )
        marker_style = _replace_with_color_overrides(
            marker_base,
            ("markerfacecolor", "markerfacealpha"),
            ("markeredgecolor", "markeredgealpha"),
            markerfacecolor=markerfacecolor,
            markerfacealpha=markerfacealpha,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgealpha=markeredgealpha,
            markeredgewidth=markeredgewidth,
        )

        self._sync_labels(
            list(scores_dict.keys()),
            add_extra_labels=add_extra_labels,
            item_name="sealevel set",
        )

        set_name = name or f"Set {len(self._sealevel_data_list) + 1}"
        self._sealevel_data_list.append(
            _SeaLevelSetData(
                name=set_name,
                scores_dict=scores_dict,
                style=line_style,
                markersettings=marker_style,
            )
        )
        self._claim_legend_if_named(name)

    @property
    def _datasets(self) -> Sequence[object]:
        return self._sealevel_data_list

    def _draw_datasets(self) -> None:
        """Draw the sealevel sets on the plot."""
        centers = self._category_centers

        # A fresh generator per build, seeded from the per-plot base seed, keeps rebuilds
        # identical in both the explicit-seed and user-generator modes.
        jitter_rng = np.random.default_rng(self._jitter_base_seed)

        for sealevel_set in self._sealevel_data_list:
            x_positions = []
            y_positions = []
            for label, value, x_center in self._present_positions(
                sealevel_set.scores_dict, centers
            ):
                horizontal_jitter = self._maximum_horizontal_jitter_per_category
                if isinstance(horizontal_jitter, dict):
                    horizontal_jitter = horizontal_jitter.get(label, 0.0)
                x_positions.append(
                    x_center + jitter_rng.uniform(low=-horizontal_jitter, high=horizontal_jitter)
                )

                vertical_jitter = self._maximum_vertical_jitter_per_category
                if isinstance(vertical_jitter, dict):
                    vertical_jitter = vertical_jitter.get(label, 0.0)
                y_positions.append(
                    value + jitter_rng.uniform(low=-vertical_jitter, high=vertical_jitter)
                )

            markersettings = sealevel_set.markersettings.to_mpl_settings_dict()
            sealevel_artists = self._ax.plot(
                x_positions,
                y_positions,
                linestyle=sealevel_set.style.linestyle,
                color=self._resolved_rgba(
                    sealevel_set.style.linecolor,
                    sealevel_set.style.linealpha,
                    field="linecolor",
                ),
                linewidth=sealevel_set.style.linewidth,
                zorder=sealevel_set.style.zorder,
                label=sealevel_set.name,
                markerfacecolor=markersettings["markerfacecolor"],
                marker=markersettings["marker"],
                markersize=markersettings["markersize"],
                markeredgecolor=markersettings["markeredgecolor"],
                markeredgewidth=markersettings["markeredgewidth"],
            )
            self._artists.track(sealevel_artists)

    def _dataset_legend_handles(self) -> list[LegendHandle]:
        """Legend handles for the sealevel sets."""
        handles: list[LegendHandle] = []

        for sealevel_data in self._sealevel_data_list:
            markersettings = sealevel_data.markersettings.to_mpl_settings_dict()
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle=cast("Any", sealevel_data.style.linestyle),
                    color=self._resolved_rgba(
                        sealevel_data.style.linecolor,
                        sealevel_data.style.linealpha,
                        field="linecolor",
                    ),
                    linewidth=sealevel_data.style.linewidth,
                    label=sealevel_data.name,
                    zorder=sealevel_data.style.zorder,
                    markerfacecolor=markersettings["markerfacecolor"],
                    marker=markersettings["marker"],
                    markersize=markersettings["markersize"],
                    markeredgecolor=markersettings["markeredgecolor"],
                    markeredgewidth=markersettings["markeredgewidth"],
                )
            )

        return handles

    def format_ylabels_as_fractions(
        self,
        denominator: int | np.integer,
        *,
        minimum_numerator: int = 0,
        maximum_numerator: int | None = None,
    ) -> None:
        """Format y-axis labels as fractions out of the given denominator.

        Args:
            denominator (int): The denominator to use for the fractions.
            minimum_numerator (int, optional): The minimum numerator to display. Defaults to 0.
            maximum_numerator (int | None, optional): The maximum numerator to display.
                Defaults to None, which uses the denominator as the maximum numerator.

        Returns:
            None

        Raises:
            TypeError: If a numerator or denominator is not an integer.
            ValueError: If bounds are inconsistent or ``denominator`` is not positive.
        """
        if not isinstance(denominator, Integral) or isinstance(denominator, (bool, np.bool_)):
            raise TypeError("denominator must be an integer.")
        denominator = int(denominator)
        if denominator <= 0:
            raise ValueError("denominator must be a positive integer.")

        original_maximum = maximum_numerator
        if maximum_numerator is None:
            maximum_numerator = denominator
        if not isinstance(minimum_numerator, int):
            raise TypeError("minimum_numerator must be an integer.")
        if not isinstance(maximum_numerator, int):
            raise TypeError("maximum_numerator must be an integer.")
        if maximum_numerator > denominator:
            raise ValueError("maximum_numerator cannot exceed denominator.")
        if minimum_numerator > maximum_numerator:
            if original_maximum is None:
                raise ValueError(
                    "minimum_numerator cannot exceed denominator when maximum_numerator is not set."
                )
            raise ValueError("minimum_numerator cannot exceed maximum_numerator.")

        self.set_yticks(
            [n / denominator for n in range(minimum_numerator, maximum_numerator + 1)],
            labels=[f"{n}/{denominator}" for n in range(minimum_numerator, maximum_numerator + 1)],
        )
