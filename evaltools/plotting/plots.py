import matplotlib.pyplot as plt
import numpy as np
import random
from .colors import *

LABEL_SIZE = 24
TICK_SIZE = 12
FIG_SIZE = (12,6)

def get_bins_and_labels(val_range, unique_vals, num_labels=8):
    """
    Get necessary information for histograms. If we're working with only a few discrete, floating point values, then
    set the bin width to be relatively thin, Otherwise, adaptively set the bin width to the scale of our data. In
    both cases, shift the tick labels over to be in the center of the bins (shift by bin_width / 2).
    TODO: clean this up..., document parameters.
    """
    if type(val_range[1]) is not int and len(unique_vals) <= 20:
        sorted_vals = sorted(unique_vals)
        bin_width = 0.2*(sorted_vals[1] - sorted_vals[0])
        hist_bins, tick_bins, tick_labels = [], [], []
        for val in sorted_vals:
            hist_bins.append(val - bin_width/2)
            hist_bins.append(val + 3*bin_width/2)
            tick_bins.append(val + bin_width/2)
            num = round(val * self.num_districts) # TODO: this is not going to work
            tick_labels.append(f"{num}/{self.num_districts}")
    else:
        bin_width = 10 ** (np.floor(np.log10(val_range[1] - val_range[0])) - 1)
        if bin_width == 0.01: # TODO: is there a cleaner way to do this...
            bin_width /= 5
        if bin_width == 0.1:
            bin_width = 1
        if bin_width >= 1:
            bin_width = int(bin_width)
        hist_bins = np.arange(val_range[0], val_range[1] + 2 * bin_width, bin_width)
        label_interval = max(int(len(hist_bins) / num_labels), 1)
        tick_bins, tick_labels = [], []
        for i, x in enumerate(hist_bins[:-1]):
            if i % label_interval == 0:
                tick_labels.append(x)
                tick_bins.append(x + bin_width / 2)
        for i, label in enumerate(tick_labels):
            if type(label) == np.float64:
                tick_labels[i] = round(label, 2)
    return hist_bins, tick_bins, tick_labels, bin_width

def draw_arrow(ax, text, orientation, padding=0.1):
    """
    For some partisan metrics, we want to draw an arrow showing where the POV-party's advantage is.
    Depending on the orientation of the scores (histograms have scores arranged horizontally, violinplots
    have scores arranged vertically), we either place the arrow at the bottom left, pointing rightward,
    or in the middle of the y-axis, pointing up.
    """
    if orientation == "horizontal":
        x = ax.get_xlim()[0]
        y = ax.get_ylim()[0] - padding*ax.get_ylim()[1]
        horizontal_align = "left"
        rotation = 0
    elif orientation == "vertical":
        x = ax.get_xlim()[0] -  padding*(sum(map(lambda x: abs(x), ax.get_xlim())))
        y = sum(ax.get_ylim())/2
        horizontal_align = "center"
        rotation = 90
    ax.text(x, y,
            text,
            ha=horizontal_align,
            va="center",
            color="white",
            rotation=rotation,
            size=10,
            bbox=dict(
                boxstyle="rarrow,pad=0.3",
                fc=defaultGray,
                alpha=1,
                ec="black",
                )
            )
    return

def plot_histogram(ax, scores, proposed_info={}):
    """
    Plot a histogram with the ensemble scores in bins and the proposed plans' scores as vertical lines.
    If there are many unique values, use a white border on the bins to distinguish, otherwise reduce the
    bin width to 80%.

    Parameters:
    ----------
        - ax: matplotlib Axis
        - scores: {str: [int]} with keys of `ensemble`, `citizen`, `proposed`
        - proposed_info: {str: [str]} with keys of `colors`, `names`
    """
    all_scores = scores["ensemble"] + scores["citizen"] + scores["proposed"]
    score_range = (min(all_scores), max(all_scores))
    hist_bins, tick_bins, tick_labels, bin_width = get_bins_and_labels(score_range, set(all_scores))
    ax.set_xticks(tick_bins)
    ax.set_xticklabels(tick_labels, fontsize=TICK_SIZE)
    rwidth    = 0.8     if len(set(scores)) < 20 else 1
    edgecolor = "black" if len(set(scores)) < 20 else "white"
    alpha = 0.7 if scores["ensemble"] and scores["citizen"] else 1
    for kind in ["ensemble", "citizen"]:
        if scores[kind]:
            ax.hist(scores[kind],
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
            ax.axvline(s + bin_width / 2 + jitter,
                        color=proposed_info['colors'][i],
                        lw=2,
                        label=f"{proposed_info['names'][i]}: {round(s,2)}",
                        )
        ax.legend()
    ax.get_yaxis().set_visible(False)
    return ax