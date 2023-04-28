from typing import List, Tuple, Union

import numpy as np
from numpy import array


def bins(scores, width=None, labels=8) -> Tuple[array, List, List, Union[float, int]]:
    """
    Get necessary information for histograms. If we're working with only a few
    discrete, floating point values, then set the bin width to be relatively thin.
    Otherwise, adaptively set the bin width to the scale of our data.

    Args:
        scores (list): The collection of all observations.
        width (int, optional): The width of the bins.
        labels (int, optional): The number of histograms to be labeled.

    Returns:
        A tuple consisting of the histogram bins, the bins that are ticked, the
        labels for the bins that are ticked, and the bin width.
    """
    # Get the minimum score and maximum score
    minscore, maxscore = min(scores), max(scores)

    # Calculate bin width using Gabe's logarithmic heuristic
    # TODO: Test this with real score data and see how it looks
    if not width:
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
        if isinstance(label, np.float64):
            tick_labels[i] = round(label, 2)

    return hist_bins, tick_bins, tick_labels, width
