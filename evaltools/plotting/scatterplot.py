from matplotlib.axes import Axes
import numpy as np

def scatterplot(ax, x, y, labels, limits=set(), proposed_info={}, resolution=50, axis_range=None) -> Axes:
    """
    Plot a scatterplot comparing two scores, with the proposed plans'
    scores as points.

    Args:
        ax (Axes): `Axes` object on which the histogram is plotted.
        x (list): Score on the x-axis.
        y (list): score on the y-axis.
        limits: (tuple, optional): x- and y-axis limits.
        proposed_info (dict, optional): Dictionary with keys of `colors`, `names`, 
            `x`, `y`; the \(i\)th color in `color` corresponds to the \(i\)th name 
            in `names`, which corresponds to the \(i\)th value in `x` and `y`.

    Returns:
        Axes object on which the scatterplot is plotted.
    """
    min_val = min(min(x), min(y))
    x_range = (min(x), max(x))
    y_range = (min(y), max(y))
    h, xedges, yedges, image = ax.hist2d(x,
                                         y,
                                         bins=[np.arange(min(x), max(x)+1, 10), np.arange(min(y), max(y)+1)],
                                         cmap='Greys',
                                        #  cmin=min_val,
                                         range=axis_range,
                                         )
    if proposed_info:
        for i in range(len(proposed_info['names'])):
            x = proposed_info['x'][i]
            y = proposed_info['y'][i]
            ax.scatter(
                    x + 0.5,
                    y,
                    # label=f"{proposed_info['names'][i]} ({x}, {y})",
                    label=f"{proposed_info['names'][i]}",
                    color=proposed_info['colors'][i],
                    s=150,
                    edgecolor='black',
                   )
    ax.legend()
    xticklabels = ax.get_xticklabels()
    xticks = ax.get_xticks()
    
    # ax.set_xticks([30.5, 35.5, 40.5, 45.5, 50.5])
    # ax.set_xticklabels(["30", "35", "40", "45", "50"])
    # ax.set_xlabel(labels[0], fontsize=20)
    # ax.set_ylabel(labels[1], fontsize=20)
    if limits:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
    return ax