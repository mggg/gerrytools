from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes import Axes

from .histogram import histogram
from .scatterplot import scatterplot


def multidimensional(
    x,
    y,
    hist,
    labels=["X values", "Y values", "Histogram values"],
    bin_width=1,
    limits=None,
    proposed_info={},
    figsize=(12, 8),
) -> Tuple[Axes, Axes]:
    r"""
    Plot a multidimensional figure, comparing two metrics as a scatterplot above
    and one metric as a histogram, below.

    Args:
        ax (Axes): `Axes` object on which the histogram is plotted.
        x (list): Score on the x-axis of the scatterplot.
        y (list): Score on the y-axis of the scatterplot.
        hist (list): Score to be plotted as a histogram below.
        limits (list, optional): x, y, and histogram limits, if wanted.
        proposed_info (dict, optional): Dictionary with keys of `colors`, `names`,
            `x`, `y`, `hist`; the \(i\)th color in `color` corresponds to the
            \(i\)th name in `names`, which corresponds to the \(i\)th value in
            `x`, `y`, and `hist`.
        figsize (tuple, optional): Figure size.

    Returns:
        The scatterplot and histogram axes.
    """
    _ = plt.subplots(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    scatter_limits = limits[:2] if limits else set()
    scatter_labels = labels[:2]
    scatter_ax = plt.subplot(gs[0])
    scatter_ax = scatterplot(
        scatter_ax,
        x,
        y,
        labels=scatter_labels,
        limits=scatter_limits,
        proposed_info=proposed_info,
    )

    scores = {
        "ensemble": hist,
        "citizen": [],
        "proposed": proposed_info["hist"] if proposed_info else [],
    }

    hist_limits = limits[-1] if limits else set()
    hist_label = labels[-1]
    hist_ax = plt.subplot(gs[1])
    hist_ax = histogram(
        hist_ax,
        scores,
        label=hist_label,
        limits=hist_limits,
        proposed_info=proposed_info,
        bin_width=bin_width,
    )

    hist_ax.get_yaxis().set_visible(False)
    hist_ax.spines["top"].set_visible(False)
    hist_ax.spines["right"].set_visible(False)
    hist_ax.spines["left"].set_visible(False)

    return scatter_ax, hist_ax
