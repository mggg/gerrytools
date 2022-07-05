from matplotlib.axes import Axes
import numpy as np


def scatterplot(ax, x, y, labels=None, limits=set(), proposed_info={}, bins=None, axis_range=None) -> Axes:
    r"""
    Plot a scatterplot comparing two scores, with the proposed plans'
    scores as points.

    Args:
        ax (Axes): `Axes` object on which the histogram is plotted.
        x (list): Score on the x-axis.
        y (list): score on the y-axis.
        labels (list, optional): Strings for x- and y-axis labels.
        limits (tuple, optional): Axis limits (specify to force plot to extend to
            these limits).
        proposed_info (dict, optional): Dictionary with keys of `colors`, `names`,
            `x`, `y`; the \(i\)th color in `color` corresponds to the \(i\)th name
            in `names`, which corresponds to the \(i\)th value in `x` and `y`.

    Returns:
        Axes object on which the scatterplot is plotted.
    """
    if not bins:
        bins = [
            np.arange(int(min(x)), int(max(x)) + 1),
            np.arange(int(min(y)), int(max(y)) + 1)
        ]

    h, xedges, yedges, image = ax.hist2d(
        x, y, bins=bins, cmap='Greys', range=axis_range,
    )

    # Shift bins over by 0.5 to center labels in the middle of the bin.
    # TODO: This only works for bins of width 1 — need to fix for general bid width.
    ax.set_xticks([x + 0.5 for x in xedges])
    ax.set_xticklabels(xedges)
    ax.set_yticks([y + 0.5 for y in yedges])
    ax.set_yticklabels(yedges)

    if proposed_info:
        for i in range(len(proposed_info['names'])):
            x = proposed_info['x'][i]
            y = proposed_info['y'][i]
            ax.scatter(
                x + 0.5,
                y + 0.5,
                label=f"{proposed_info['names'][i]} ({x}, {y})",
                color=proposed_info['colors'][i],
                s=150,
                edgecolor='black',
            )
        ax.legend()

    if labels:
        ax.set_xlabel(labels[0], fontsize=24)
        ax.set_ylabel(labels[1], fontsize=24)
    if limits:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
    return ax
