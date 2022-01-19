
from matplotlib.axes import Axes
import random
from .bins import bins
from .colors import defaultGray, citizenBlue


def histogram(ax, scores, label="Histogram value", limits=set(), proposed_info={}, ticksize=12, bin_width=None) -> Axes:
    """
    Plot a histogram with the ensemble scores in bins and the proposed plans'
    scores as vertical lines. If there are many unique values, use a white border
    on the bins to distinguish, otherwise reduce the bin width to 80%.

    TODO: refactor `proposed_info` later to use more python builtin tools.

    Args:
        ax (Axes): `Axes` object on which the histogram is plotted.
        scores (dict): Dictionary with keys of `ensemble`, `citizen`, `proposed`
            which map to lists of numerical scores.
        proposed_info (dict, optional): Dictionary with keys of `colors`, `names`;
            the \(i\)th color in `color` corresponds to the \(i\)th name in `names`.
        limits: (tuple, optional): x-axis limits.
        ticksize (float, optional): Font size of tick labels.

    Returns:
        Axes object on which the histogram is plotted.
    """
    # Put all scores into a single list.
    all_scores = scores["ensemble"] + scores["citizen"] + scores["proposed"]
    if not bin_width:
        # Get the necessary bins, ticks, labels, and bin width.
        hist_bins, tick_bins, tick_labels, bin_width = bins(set(all_scores).union(limits))
    else:
        hist_bins, tick_bins, tick_labels, bin_width = bins(set(all_scores).union(limits), bin_width)

    # Set xticks and xticklabels.
    ax.set_xticks(tick_bins)
    ax.set_xticklabels(tick_labels, fontsize=ticksize)

    # Adjust the visual width of the bins according to the number of observations;
    # if we have few scores, we want to adjust the look of the bins to make the
    # plots more readable. Also adjust the opacity of the ensembles if we include
    # a citizen ensemble.
    rwidth    = 0.8     if len(set(scores)) < 20 else 1
    edgecolor = "black" if len(set(scores)) < 20 else "white"
    alpha = 0.7 if scores["ensemble"] and scores["citizen"] else 1

    for kind in ["ensemble", "citizen"]:
        if scores[kind]:
            ax.hist(
                scores[kind],
                bins=hist_bins,
                color=defaultGray if kind == "ensemble" else citizenBlue,
                rwidth=rwidth,
                edgecolor=edgecolor,
                alpha=alpha,
                density=True,
            )
    if scores["proposed"]:
        for i, s in enumerate(scores["proposed"]):
            jitter = random.uniform(-bin_width/5, bin_width/5) if scores["proposed"].count(s) > 1 else 0

            # Plot vertical line.
            ax.axvline(
                s + bin_width / 2 + jitter,
                color=proposed_info['colors'][i],
                lw=2,
                label=f"{proposed_info['names'][i]}: {round(s,2)}",
            )
        
        ax.legend()
    ax.set_xlabel(label, fontsize=24)
    ax.get_yaxis().set_visible(False)
    if limits:
        ax.set_xlim(limits)
    return ax
