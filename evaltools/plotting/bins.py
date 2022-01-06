
import numpy as np
from numpy import array
from typing import Tuple, List, Union


def bins(scores, labels=8) -> Tuple[array, List, List, Union[float, int]]:
    """
    Get necessary information for histograms. If we're working with only a few
    discrete, floating point values, then set the bin width to be relatively thin.
    Otherwise, adaptively set the bin width to the scale of our data.
    
    Args:
        scores (list): The collection of all observations.
        labels (int, optional): The number of histograms to be labeled.

    Returns:
        A tuple consisting of the histogram bins, the bins that are ticked, the
        labels for the bins that are ticked, and the bin width.
    """
    # Get the minimum score, maximum score, 25th and 7th percentiles, and the IQR
    # of the observations.
    minscore, maxscore = min(scores), max(scores)
    l, r = np.percentile(list(scores), [25, 75])
    iqr = r-l
    n = len(scores)

    # Calculate the bin width using the Freedman-Diaconis rule; if all observations
    # are integers, round the bin width to the nearest integer.
    allints = all(type(score) is int for score in scores)
    fdr = 2*iqr*n**(-1/3)
    # width = round(fdr) if allints else fdr

    # Calculate bin width using Gabe's logarithmic rule
    # TODO: Test this with real score data and see how it looks
    width = 10 ** (np.floor(np.log10(maxscore - minscore)) - 1)
    if width == 0.01:
        width /= 5
    if width == 0.1:
        width = 1
    if width >= 1:
        width = int(width)

    hist_bins = np.arange(minscore, maxscore + 2 * width, width)
    label_interval = max(int(len(hist_bins) / labels), 1)
    tick_bins, tick_labels = [], []
    for i, x in enumerate(hist_bins[:-1]):
        if i % label_interval == 0:
            tick_labels.append(x)
            tick_bins.append(x + width / 2)
    for i, label in enumerate(tick_labels):
        if type(label) == np.float64:
            tick_labels[i] = round(label, 2)
    
    return hist_bins, tick_bins, tick_labels, width
