from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection

from gerrytools.logging import get_logger
from gerrytools.plotting.data._categorical_distribution_base import CategoricalDistributionPlotBase
from gerrytools.plotting.data.options import ViolinPlotOptions
from gerrytools.plotting.utils import UNSET, Unset
from gerrytools.typing import Color, LegendHandle

logger = get_logger(__name__)


@dataclass(frozen=True)
class _ViolinPlotSetData:
    """A set of violinplots to render for one model/series."""

    name: str
    scores_dict: dict[str, list[float]]
    style: ViolinPlotOptions


class ViolinPlot(CategoricalDistributionPlotBase):
    """Create grouped violinplot comparison figures across categories."""

    _dataset_noun = "violinplot set"

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
        """Initialize a ViolinPlot.

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
            width_scale (float, optional): Relative width of violins within each slot. Defaults to
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
        self._violinplot_data_list: list[_ViolinPlotSetData] = []

    def add_dataset(
        self,
        scores: dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame,
        name: str | None = None,
        *,
        category_labels: list[str] | None = None,
        options: ViolinPlotOptions | None = None,
        facecolor: Color | None | Unset = UNSET,
        facealpha: float | None = None,
        edgecolor: Color | None | Unset = UNSET,
        edgealpha: float | None = None,
        edgewidth: float | None = None,
        add_extra_labels: bool = False,
        zorder: int | None = None,
    ) -> None:
        """Add one violinplot dataset (one violin per category) to the figure.

        Non-finite samples are dropped per category; a category with no finite samples
        draws no violin but keeps its axis slot.

        Args:
            scores (dict[str, list[float]] | list[float] | list[list[float]] | pd.DataFrame):
                Distribution input by category.
            category_labels (list[str] | None, optional): Labels for list-based input.
                Defaults to None.
            name (str | None, optional): Legend label for the dataset. Defaults to None.
            options (ViolinPlotOptions | None, optional): Base styling whose values are used
                for any styling argument left unset. Defaults to None.
            facecolor (Color | None, optional): Violin fill color. Pass ``None`` for an
                unfilled violin. Omit to use the ``options`` default ``"default_grey"``.
            facealpha (float | None, optional): Violin fill alpha override. Defaults to None.
            edgecolor (Color | None, optional): Violin edge color. Pass ``None`` for no edge.
                Omit to use the ``options`` default ``"black"``.
            edgealpha (float | None, optional): Violin edge alpha override. Defaults to None.
            edgewidth (float, optional): Violin edge width. Defaults to ``0.8``.
            add_extra_labels (bool, optional): Whether to merge unseen incoming labels into
                existing category labels. Defaults to False.
            zorder (int, optional): Draw order for the dataset. Defaults to ``1``.

        Returns:
            None

        Raises:
            ValueError: If ``scores`` is empty or its labels or styling options are invalid.
        """
        base = options if options is not None else ViolinPlotOptions()
        style = base.merged(
            facecolor=facecolor,
            facealpha=facealpha,
            edgecolor=edgecolor,
            edgealpha=edgealpha,
            edgewidth=edgewidth,
            zorder=zorder,
        )

        scores_dict = self._convert_distribution_data_to_dictionary(scores, category_labels)
        if len(scores_dict) == 0:
            raise ValueError("scores is empty; provide scores for at least one category.")
        self._sync_labels(
            list(scores_dict.keys()),
            add_extra_labels=add_extra_labels,
            item_name="violinplot set",
        )

        set_name = name or f"Set {len(self._violinplot_data_list) + 1}"
        self._violinplot_data_list.append(
            _ViolinPlotSetData(scores_dict=scores_dict, name=set_name, style=style)
        )
        self._claim_legend_if_named(name)

    @property
    def _datasets(self) -> Sequence[object]:
        return self._violinplot_data_list

    def _draw_datasets(self) -> None:
        """Draw the violinplots on the plot."""
        centers, offsets, widths = self._grouped_layout(len(self._violinplot_data_list))

        for k, violinplot_data in enumerate(self._violinplot_data_list):
            data_k: list[list[float]] = []
            pos_k: list[float] = []
            for _label, vals, x in self._present_positions(
                violinplot_data.scores_dict, centers + offsets[k]
            ):
                # Non-finite samples would silently empty the violin's KDE body.
                finite_vals = np.asarray(vals, dtype=float)
                finite_vals = finite_vals[np.isfinite(finite_vals)]
                if finite_vals.size == 0:
                    continue
                data_k.append(finite_vals.tolist())
                pos_k.append(x)

            if len(data_k) == 0:
                continue

            vp = self._ax.violinplot(
                data_k,
                positions=pos_k,
                widths=widths,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            # With showmeans/medians/extrema disabled, the returned dict carries only bodies.
            style = violinplot_data.style
            bodies = cast(list[PolyCollection], vp.get("bodies", []))
            for patch in bodies:
                self._artists.track(patch)
                patch.set_alpha(None)
                patch.set_facecolor(
                    self._resolved_rgba(style.facecolor, style.facealpha, field="facecolor")
                )
                patch.set_linewidth(style.edgewidth)
                patch.set_edgecolor(
                    self._resolved_rgba(style.edgecolor, style.edgealpha, field="edgecolor")
                )
                patch.set_zorder(style.zorder)

    def _dataset_legend_handles(self) -> list[LegendHandle]:
        """Legend handles for the violinplot sets."""
        return self._patch_legend_handles(
            (violin_data.name, violin_data.style) for violin_data in self._violinplot_data_list
        )
