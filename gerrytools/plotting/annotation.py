from matplotlib.axes import Axes

from .colors import defaultGray


def arrow(ax, text, orientation="horizontal", color=defaultGray, padding=0.1) -> Axes:
    """
    For some partisan metrics, we want to draw an arrow showing where the POV-party's
    advantage is. Depending on the orientation of the scores (histograms have
    scores arranged horizontally, violinplots have scores arranged vertically),
    we either place the arrow at the bottom left, pointing rightward, or in the
    middle of the y-axis, pointing up.

    Args:
        ax (Axes): `Axes` object onto which the arrow's plotted.
        text (str): String plotted on top of the arrow.
        orientation (str, optional): Direction the arrow's pointing; acceptable
            values are `"horizontal"` and `"vertical"`. Defaults to `"horizontal"`.
        color (str, optional): Color of the arrow.
        padding (float, optional): Spacing between the arrow and its axis. Defaults
            to `0.1`.

    Returns:
        matplotlib `Axes`.
    """

    if orientation == "horizontal":
        x = ax.get_xlim()[0]
        y = ax.get_ylim()[0] - padding * ax.get_ylim()[1]
        horizontal_align = "left"
        rotation = 0
    elif orientation == "vertical":
        x = ax.get_xlim()[0] - padding * (sum(map(lambda x: abs(x), ax.get_xlim())))
        y = sum(ax.get_ylim()) / 2
        horizontal_align = "center"
        rotation = 90

    ax.text(
        x,
        y,
        text,
        ha=horizontal_align,
        va="center",
        color="white",
        rotation=rotation,
        size=10,
        bbox=dict(
            boxstyle="rarrow,pad=0.3",
            fc=color,
            alpha=1,
            ec="black",
        ),
    )

    return ax


def ideal(ax, label, placement, orientation, color=defaultGray, alpha=0.1):
    """
    Adds a vertical line, horizontal line, or band indicating the ideal value
    (or range of values) for the provided score.

    Args:
        ax (Axes): `Axes` object onto which the line's plotted.
        label (str): Label for the ideal score.
        placement (float,tuple): If plotting a line, a single value; if plotting
            a band, a tuple of (start, end) values.
        orientation (str): Indicates the direction of the line or band. Acceptable
            values are `"horizontal"` or `"vertical"`.
        color (str, optional): Color of the line or band. Defaults to `defaultGray`.
        alpha (float, optional): Opacity of the line or band. Defaults to `0.1`.
    """
    orig_xlims = ax.get_xlim()
    orig_ylims = ax.get_ylim()

    # Warn the user and abort if the `placement` parameter isn't of the correct
    # type.
    if type(placement) not in {float, int, tuple}:
        raise TypeError("`placement` is not of correct type.")

    # If `placement` is a tuple, we draw a band.
    if isinstance(placement, tuple):
        if orientation == "horizontal":
            xlims = orig_xlims
            ylims1 = [placement[0], placement[0]]
            ylims2 = [placement[1], placement[1]]
        elif orientation == "vertical":
            xlims = placement
            ylims1 = [orig_ylims[0], orig_ylims[0]]
            ylims2 = [orig_ylims[1], orig_ylims[1]]

        ax.fill_between(xlims, ylims1, ylims2, color=color, alpha=alpha, label=label)
    # Otherwise, draw a line.
    else:
        alpha = alpha if alpha else 0.5
        idealprops = dict(color=color, alpha=alpha, label=label)

        if orientation == "horizontal":
            ax.axhline(placement + 0.5, **idealprops)
        else:
            ax.axvline(placement + 0.5, **idealprops)

    # Set the original x- and y-axis limits, and plot a legend.
    ax.set_xlim(orig_xlims)
    ax.set_ylim(orig_ylims)
    ax.legend()

    return ax
