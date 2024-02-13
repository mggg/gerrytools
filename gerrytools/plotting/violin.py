import random

import numpy as np
from matplotlib.axes import Axes

from .colors import citizenBlue, defaultGray, districtr


def violin(
    ax,
    scores,
    xticklabels=None,
    labels=None,
    proposed_info={},
    percentiles=(1, 99),
    rotation=0,
    ticksize=12,
    jitter=1 / 3,
) -> Axes:
    r"""
    Plot a violin plot, which takes `scores` — a dictionary where each value (corresponding to
    an ensemble, citizens' ensemble, or proposed plans), will be a list of lists, where each
    sublist will be its own violin. Proposed scores will be plotted as colored circles on
    their respective violin. Color the violins conditioned on the kind of the scores
    (ensemble or citizen), and trim each sublist to only the values between the percentiles.

    Args:
        ax (Axes): `Axes` object on which the violins are plotted.
        scores (dict): Dictionary with keys of `ensemble`, `citizen`, `proposed`
            which map to lists of numerical scores.
        proposed_info (dict, optional): Dictionary with keys of `colors`, `names`;
            the \(i\)th color in `color` corresponds to the \(i\)th name in `names`.
        percentiles (tuple, optional): Observations outside this range of
            percentiles are ignored. Defaults to `(1, 99)`, such that observations
            between the 1st and 99th percentiles (inclusive) are included, and
            all others are ignored.
        rotation (float, optional): Tick labels are rotated `rotation` degrees
            _counterclockwise_.
        ticksize (float, optional): Font size for tick labels.
        jitter (float, optional): When there is more than one proposed plan,
            adjust its detail points by a value drawn from \(\mathcal U (-\epsilon,
            \epsilon)\) where \(\epsilon = \) `jitter`.
        labels (list, optional): x- and y-axis labels, if desired.
        xticklabels (list, optional): Labels for the violins, default to integers.

    Returns:
        `Axes` object on which the violins are plotted.
    """
    # Get all the scores into one list; pick a face color.
    ensemble = scores["ensemble"] if scores["ensemble"] else scores["citizen"]
    facecolor = defaultGray if scores["ensemble"] else citizenBlue

    # Initialize a list for winnowing scores.
    trimmed_scores = []

    # Pare each ensemble down to only the observations between the 1st and 99th
    # percentiles.
    for score_list in ensemble:
        low = np.percentile(ensemble, percentiles[0])
        high = np.percentile(ensemble, percentiles[1])
        # print(f"Only including scores between [{low}, {high}]")
        trimmed_scores.append([s for s in score_list if s >= low and s <= high])

    # Plot violins.
    parts = ax.violinplot(trimmed_scores, showextrema=False)

    # For each of the violins, modify its visual properties; change the face color
    # to the specified face color, change its edge color to black, and set its
    # opacity to 1.
    for pc in parts["bodies"]:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor("black")
        pc.set_alpha(1)

    # Set xticks, xlabels, and x-axis limits.
    if not xticklabels:
        xticklabels = range(1, len(scores["ensemble"]) + 1)
    ax.set_xticks(range(1, len(ensemble) + 1))
    ax.set_xticklabels(xticklabels, fontsize=ticksize, rotation=rotation)
    ax.set_xlim(0.5, len(ensemble) + 0.5)

    # Plot each proposed plan individually, adjusting its detail points by
    # a value drawn from the uniform distribution of specified width centered on
    # the index of the violin.
    if scores["proposed"]:
        for violin in range(len(scores["proposed"])):
            for plan, score in enumerate(scores["proposed"][violin]):
                # Horizontally jitter proposed scores if there are multiple scores
                # at the same height.
                jitter_val = (
                    random.uniform(-jitter, jitter)
                    if scores["proposed"][violin].count(score) > 1
                    else 0
                )
                ax.scatter(
                    violin + 1 + jitter_val,
                    score,
                    color=(
                        scores["proposed"]["colors"][violin]
                        if not isinstance(scores["proposed"], list)
                        and scores["proposed"]["colors"]
                        else districtr(plan + 1).pop()
                    ),
                    edgecolor="black",
                    s=100,
                    alpha=0.9,
                    label=proposed_info["names"][plan] if violin == 0 else None,
                )
        ax.legend()
        ax.grid(axis="x")

    if labels:
        ax.set_xlabel(labels[0], fontsize=24)
        ax.set_ylabel(labels[1], fontsize=24)

    return ax
