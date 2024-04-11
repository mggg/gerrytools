import numpy as np
from matplotlib.axes import Axes

from .colors import districtr


def scatterplot(
    ax,
    x,
    y,
    labels=None,
    limits=set(),
    bins=None,
    axis_range=None,
    show_legend=True,
) -> Axes:
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
        show_legend (bool, optional): If `True`, show the legend. Generally helpful
            when trying to distinguish relationships between blocs within congressional
            districts, but can be cumbersome when there are many districts (e.g., 20+).
            Defaults to `True`.
    Returns:
        Axes object on which the scatterplot is plotted.
    """
    for x_score, y_score in zip(x, y):
        for i in range(len(x_score)):
            x_val = x_score[i]
            y_val = y_score[i]
            ax.scatter(
                x_val + 0.5,
                y_val + 0.5,
                color=f"{districtr(i+1).pop()}",
                s=150,
                edgecolor="black",
                label=f"{i+1}",
            )
    if show_legend:
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
    xedges = np.arange(x_min, x_max + 1, 1)
    yedges = np.arange(y_min, y_max + 1, 1)
    # TODO: This only works for bins of width 1 — need to fix for general bid width.
    ax.set_xticks([x + 0.5 for x in xedges])
    ax.set_xticklabels(xedges)
    ax.set_yticks([y + 0.5 for y in yedges])
    ax.set_yticklabels(yedges)

    return ax


# def scatterplot(
#     ax,
# 	x_lst,
# 	y_lst,
# 	labels=None,
# 	limits=set(),
# 	bins=None,
# 	axis_range=None
# ) -> Axes:
#     r"""
#     Plot a scatterplot comparing two scores, with the proposed plans'
#     scores as points.

#     Args:
#         ax (Axes): `Axes` object on which the histogram is plotted.
#         x (list): Score on the x-axis. This will be a list of lists, where each
#            sub-list corresponds to the scores for an individual plan.
#         y (list): Score on the y-axis. This will be a list of lsits where each
#            sub-list corresponds to the scores for an individual plan.
#         labels (list, optional): Strings for x- and y-axis labels.
#         limits (tuple, optional): Axis limits (specify to force plot to extend to
#             these limits).
#         colors (list, optional): A list of colors where the ith color corresponds
#           to the ith score sub-list.
#     Returns:
#         Axes object on which the scatterplot is plotted.
#     """
#     print(x_lst)
#     print(y_lst)

#     for i, (x_score, y_score) in enumerate(zip(x_lst, y_lst)):
#         ax.scatter(
#             x_score + 0.5,
#             y_score + 0.5,
#             color=f"{districtr(i+1).pop()}",
#             s=150,
#             edgecolor="black",
#         )
#     ax.legend()

#     if labels:
#         ax.set_xlabel(labels[0], fontsize=24)
#         ax.set_ylabel(labels[1], fontsize=24)
#     if limits:
#         ax.set_xlim(limits[0])
#         ax.set_ylim(limits[1])

#     # Shift bins over by 0.5 to center labels in the middle of the bin.
#     x_min = min([min(x_point) for x_point in x_lst])
#     x_max = max([max(x_point) for x_point in x_lst])
#     y_min = min([min(y_point) for y_point in y_lst])
#     y_max = max([max(y_point) for y_point in y_lst])
#     xedges = np.arange(x_min, x_max + 1, 1)
#     yedges = np.arange(y_min, y_max + 1, 1)
#     # TODO: This only works for bins of width 1 — need to fix for general bid width.
#     ax.set_xticks([x + 0.5 for x in xedges])
#     ax.set_xticklabels(xedges)
#     ax.set_yticks([y + 0.5 for y in yedges])
#     ax.set_yticklabels(yedges)

#     return ax
