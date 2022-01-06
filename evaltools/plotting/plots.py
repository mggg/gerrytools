import matplotlib.pyplot as plt
import numpy as np
import random
from .colors import districtr

LABEL_SIZE = 24
TICK_SIZE = 12
FIG_SIZE = (12,6)
defaultGray = "#5c676f"
citizenBlue = "#4693b3"

def get_bins_and_labels(val_range, unique_vals, num_labels=8):
    """
    Get necessary information for histograms. If we're working with only a few discrete, floating point values, then
    set the bin width to be relatively thin, Otherwise, adaptively set the bin width to the scale of our data. In
    both cases, shift the tick labels over to be in the center of the bins (shift by bin_width / 2).
    TODO: clean this up... document parameters.
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

def draw_ideal(ax, label, placement, orientation, color=None, alpha=None):
    color = color if color else defaultGray
    orig_xlims = ax.get_xlim()
    orig_ylims = ax.get_ylim()

    if type(placement) is list:
        alpha = alpha if alpha else 0.1
        if orientation == "horizontal":
            xlims = orig_xlims
            ylims1 = [placement[0], placement[0]]
            ylims2 = [placement[1], placement[1]]
        elif orientation == "vertical":
            xlims = placement
            ylims1 = [orig_ylims[0], orig_ylims[0]]
            ylims2 = [orig_ylims[1], orig_ylims[1]]
        ax.fill_between(xlims,
                        ylims1,
                        ylims2,
                        color=color,
                        alpha=alpha,
                        label=label,
                       )
    else:
        alpha = alpha if alpha else 0.5
        if orientation == "horizontal":
            ax.axhline(placement + 0.5, # shifted to align with bins
                       color=color,
                       alpha=alpha, 
                       label=label,
                      )
        elif orientation == "vertical":
            ax.axvline(placement + 0.5, # shifted to align with bins
                       color=color,
                       alpha=alpha, 
                       label=label,
                      )    
    ax.set_xlim(orig_xlims)
    ax.set_ylim(orig_ylims)
    ax.legend()
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

def plot_violin(ax, scores, labels, proposed_info={}, percentiles=(1,99), tick_rotation=0, tick_size=TICK_SIZE):
    """
    Plot a violin plot, which takes `scores` — a list of lists, where each sublist will be its own violin.
    Proposed scores will be plotted as colored circles on their respective violin.
    Color the violins conditioned on the kind of the scores (ensemble or citizen), and if plotting ensemble, then
    trim each sublist to only the values between the 1-99th percentile, to match our boxplits (otherwise don't trim).

    Parameters:
    ----------
        - ax: matplotlib Axis
        - scores: {str: [int]} with keys of `ensemble`, `citizen`, `proposed`
        - proposed_info: {str: [str]} with keys of `colors`, `names`
    """
    trimmed_scores = []
    ensemble = scores["ensemble"] if scores["ensemble"] else scores["citizen"]
    facecolor = defaultGray if scores["ensemble"] else citizenBlue
    for score_list in ensemble:
        low = np.percentile(ensemble, percentiles[0])
        high = np.percentile(ensemble, percentiles[1])
        # print(f"Only including scores between [{low}, {high}]")
        trimmed_scores.append([s for s in score_list if s >= low and s <= high])
    parts = ax.violinplot(trimmed_scores, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor("black")
        pc.set_alpha(1)

    ax.set_xticks(range(1, len(ensemble) + 1))
    ax.set_xticklabels(labels, fontsize=tick_size, rotation=tick_rotation)
    ax.set_xlim(0.5, len(ensemble) + 0.5)

    if scores["proposed"]:
        for i in range(len(scores["proposed"])):
            for j, s in enumerate(scores["proposed"][i]):
                # horizontally jitter proposed scores regardless of whether there are multiple scores at the same height
                jitter = 0#random.uniform(-1/3, 1/3) #if proposed_scores[i].count(s) > 1 else 0
                ax.scatter(i + 1 + jitter,
                           s,
                           color=districtr(j+1)[-1],
                           edgecolor='black',
                           s=100,
                           alpha=0.9,
                           label=proposed_info["names"][j] if i == 0 else None,
                           )
        ax.legend()
        ax.grid(axis='x')
    return ax
