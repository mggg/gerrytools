import random

import numpy as np
from matplotlib.axes import Axes

from .colors import defaultGray
from .utils import sort_elections


def sealevel(ax, scores, num_districts, proposed_info, ticksize=12) -> Axes:
    r"""
    Plot a sea level plot: Each plan is a line across our elections on the
    x-axis, with Democratic vote share on the y-axis. The statewide Dem. vote
    share (proportionality) is plotted as a thick blue line.

    Args:
        ax (Axes): `Axes` object on which the sea level plot is plotted.
        scores (dict): Dictionary with keys of each plan plus a `statewide` key
            for proportionality. Each value is another dictionary, with keys for
            each election, values are the # seats.
        proposed_info (dict, optional): Dictionary with keys of `colors`, `names`;
            the \(i\)th color in `color` corresponds to the \(i\)th name in `names`.
        ticksize (float, optional): Font size for tick labels.
    """
    assert "statewide" in scores
    elections = sort_elections(scores["statewide"].keys())
    shares_by_plan = {plan: [] for plan in scores}
    for plan in scores:
        for election in elections:
            shares_by_plan[plan].append(scores[plan][election])

    ax.plot(
        shares_by_plan["statewide"],
        marker="o",
        markersize=10,
        lw=5,
        label="Proportionality",
    )

    for i, plan in enumerate(proposed_info["names"]):
        for j in range(len(shares_by_plan[plan])):
            if (
                len(set([shares_by_plan[plan][j] for plan in shares_by_plan.keys()]))
                > 1
            ):
                jitter = random.uniform(-0.02, 0.02)
            else:
                0

            shares_by_plan[plan][j] = shares_by_plan[plan][j] + jitter

        ax.plot(
            shares_by_plan[plan],
            marker="o",
            linestyle="--",
            color=proposed_info["colors"][i],
            label=plan,
        )

    ax.legend()

    if num_districts <= 20:
        yticks = np.arange(0, 1 + 1 / num_districts, 1 / num_districts)
        yticklabels = [f"{i}/{num_districts}" for i in range(num_districts + 1)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    ax.axhline(0.5, color=defaultGray, label="50%")
    ax.set_xticks(range(len(elections)))
    ax.set_xticklabels(elections, fontsize=ticksize)
    ax.set_ylim(-0.02, 1)

    return ax
