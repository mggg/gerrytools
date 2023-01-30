from matplotlib.axes import Axes
import numpy as np


def scatterplot(ax, x, y, labels=None, limits=set(), bins=None, axis_range=None) -> Axes:
    r"""
    Plot a scatterplot comparing two scores, with the proposed plans'
    scores as points.

    Args:
        ax (Axes): `Axes` object on which the histogram is plotted.
        x (list): Score on the x-axis. This will be a list of lists, where each
           sub-list corresponds to the scores for an individual plan.
        y (list): Score on the y-axis. This will be a list of lsits where each 
           sub-list corresponds to the scores for an individual plan.
        labels (list, optional): Strings for x- and y-axis labels.
        limits (tuple, optional): Axis limits (specify to force plot to extend to
            these limits).
        colors (list, optional): A list of colors where the ith color corresponds
          to the ith score sub-list.
    Returns:
        Axes object on which the scatterplot is plotted.
    """
    
    for x_score, y_score in zip(x, y): 
        for i in range(len(x_score)):
            x = x_score[i]
            y = y_score[i]
            ax.scatter(
                x + 0.5,
                y + 0.5,
                label=f"{names[i] if names else ''}",
                color=f"{colors[i] if colors else districtr(i).pop()},
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

    # Shift bins over by 0.5 to center labels in the middle of the bin.
    x_min = min([min(x_point) for x_point in x])
    x_max = max([max(x_point) for x_point in x])
    y_min = min([min(y_point) for y_point in y])
    y_max = max([max(y_point) for y_point in y])
    xedges = np.arange(x_min, x_max+1, 1)
    yedges = np.arange(y_min, y_max+1, 1)
    # TODO: This only works for bins of width 1 — need to fix for general bid width.
    ax.set_xticks([x + 0.5 for x in xedges])
    ax.set_xticklabels(xedges)
    ax.set_yticks([y + 0.5 for y in yedges])
    ax.set_yticklabels(yedges)

    return ax
