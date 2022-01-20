from matplotlib.axes import Axes
from .colors import defaultGray
from .utils import sort_elections
import numpy as np

def sealevel(ax, scores, num_districts, proposed_info):
    """
    Plot a sea level plot: Each plan is a line across our elections on the 
    x-axis, with Democratic vote share on the y-axis. The statewide Dem. vote
    share (proportionality) is plotted as a thick blue line.

    Args:
        ax (Axes): `Axes` object on which the sea level plot is plotted.
        scores (dict): Dictionary with keys of each plan plus a `statewide` key for proportionality.
            Each value is another dictionary, with keys for each election, values are the # seats.
        proposed_info (dict, optional): Dictionary with keys of `colors`, `names`;
            the \(i\)th color in `color` corresponds to the \(i\)th name in `names`.
    """
    assert "statewide" in scores
    elections = sort_elections(scores["statewide"].keys())
    seats_by_plan = {plan:[] for plan in scores}
    for plan in scores:
        for election in elections:
            seats_by_plan[plan].append(scores[plan][election])
    print(seats_by_plan)

    ax.plot(proportional_share,
            marker='o',
            markersize=10,
            lw=5,
            label="Proportionality",
           )
    for plan in proposed_info['names']:
        for j in range(len(seats_by_plan[plan])):
            jitter = 0 # ADD JITTER
        ax.plot(seats_by_plan[plan],
                marker='o',
                linestyle='--',
                color=proposed_info['colors'],
                label=plan,
               )
    if num_districts <= 20:
        yticks = np.arange(0, 1 + 1/num_districts, 1/num_districts)
        yticklabels = [f"{i}/{num_districts}" for i in range(num_districts + 1)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    ax.axhline(0.5, color=defaultGray, label="50%")


